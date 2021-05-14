#include <stdio.h>
#include <ctype.h>

#define THREADS 512
int BLOCKS = 0;
#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)

int MAX_LITERALS   = 8000000;  
int MAX_CLAUSES   = 550000;

int total_variables , total_clauses , total_literals;

int* formula_device;
int* formula_host;
int* clause_pointer_device;
int* clause_pointer_host;
int* assignment_device;
int* assignment_host;


//used in mask_prog
int* num_free_literals_device; //number of free_literals in the clause candidate
int* num_free_literals_host;
int* clause_candidates_device;
int* clause_candidates_host;
int* literal_candidates_device;
int* literal_candidates_host;

//for back tracking
int* level;
int current_level;
int* assignment_stack;
int assignment_stack_pointer;
int* refs;

__host__ int load_dimacs_cnf(char* filename, int* total_variables, int* total_clauses, int* total_literals ){
	int i=0;
	int j=1;
	int max = 0;
	int flag=0;
	char c;
	FILE* file;
	printf("input file %s \n",filename);
	
	clause_pointer_host[0]=0;  
	if((file = fopen(filename,"r"))==NULL) {
		printf("File %s not found \n",filename);
		exit(-1);
	}
	else 
	{ 
  	while(!feof(file))
  	{
  		c = fgetc(file);
    		if (c =='c' || c =='p' || c =='%')
    		{
       		c = fgetc(file);
       		while (c != '\r' && c != '\n')
     				c = fgetc(file);
    		}
    		else if (c=='-'){       
       		flag=1;
       		int sum = 0;
       		c = fgetc(file);
       		while(isdigit(c))
       		{
       			sum=sum*10 + c - '0';
       			c = fgetc(file);
       		}
			formula_host[i]=-sum; 
       		max = MAX(max,-formula_host[i]); 
       		i++;
      	}      
      	else if (isdigit(c) && c!='0'){
       		flag=1;
       		int sum = c - '0';
       		c = fgetc(file);
       		while(isdigit(c))
       		{
       			sum=sum*10 + c - '0';
       			c = fgetc(file);
       		}
       		formula_host[i] =sum;
       		max = MAX(max,formula_host[i]); 
       		i++;
      	} 
      	else if ( c == '0' && flag==1 ){
      		clause_pointer_host[j]=i;
      		j++;
      		flag=0;
      	}   
  	}            
  	*total_variables = max;   
  	*total_clauses = j-1;    
  	*total_literals = i;
  	printf("total number of clauses: %d, total number of variables: %d, total number of literals: %d \n",*total_clauses,*total_variables,*total_literals);
  	fclose(file);     
	}  
  return 0;
}


__global__ void mask_prop_gpu(int* dev_free_var,
                              int* dev_clause_cand,
                              int* dev_lit_cand,
                              int* formula,
                              int* assigned_vars, 
                              int* clause_ptrs,
                              int total_clauses,
                              int total_literals,
                              int total_variables )
{
  /* GPU kernel to handle parallelizable part of figuring out satisfiability */

  /* Calculate satisfiability for each clause.
   * We run in parallel, strided by how many threads we have. */

  const int stride = blockDim.x * gridDim.x;
  int clause_num = blockDim.x * blockIdx.x + threadIdx.x;
  const int thread_num = threadIdx.x;

  /* We want to keep track of which variable would be the best candidate to assign a value to,
   * and also keep track of which clause that direcly pertains to. */
  __shared__ int clause_candidate[THREADS]; /* Clause index of the best candidate */
  __shared__ int literal_candidate[THREADS]; /* Literal index of the best literal */
  __shared__ int free_var[THREADS]; /* Number of free variables for a particular clause */

  clause_candidate[thread_num]=0;
  literal_candidate[thread_num]=0;
  free_var[thread_num]=0;
  /*
  if(clause_num==0)
  {

    printf("formula: \n");
  for(int i =0 ; i < total_literals;i++){
    printf("%d ", formula[i]);
  }printf("\n");
    printf("clause pointer: \n");
  for(int i =0 ; i < total_clauses; i++){
    printf("%d ", clause_ptrs[i]);
  }
  printf("\n");
  printf("assigned_vars: \n");
  for(int i =1 ; i < total_variables+1; i++){
    printf("%d ", assigned_vars[i]);
  }
  printf("\n");
  }
  */
  while (clause_num < total_clauses) {
    /* Get start and ending literal indices for the current clause */
    int start = clause_ptrs[clause_num];
    int end = clause_ptrs[clause_num + 1];
    
    int sat = -1;
    int num_free = 0;
    
    int best_literal = 0x7FFFFFFF; /* Hopefully we don't run into an example with 2^32-1 variables */
    for (int i = start; i < end; i++) {
      int literal = formula[i];
      int value = assigned_vars[abs(literal)]; /* Take abs value in order to turn literal into variable reference */
      if ((literal > 0 && value == 1) || (literal < 0 && value == 0)) {
        /* Evaluated value of literal is equal to 1 (true). In this case, we don't care about any other literals since
         * no matter what value they have, the clause is already satisfied. */
        sat = 1;
        num_free = 0;

        break;
      } else if (value == -1) {
        /* If the literal has not been assigned a value yet, it is a free variable. We want to keep track of it. */
        num_free++;
        if (abs(literal) < abs(best_literal)) {
           /* We give preference to variables of smaller number */
           best_literal = literal;
        }
        sat = 0;
      }
    }
    if (sat == -1) {
      num_free = -1;
    }
    
    /* Update array for best values per thread: */
    if (num_free == -1 || free_var[thread_num] == -1) {

      /* Current or previous clause is NSAT */
      if (num_free == -1 && free_var[thread_num] != -1) {
        /* New clause is NSAT, conflicts with later clauses. */
        free_var[thread_num] = num_free;
        clause_candidate[thread_num] = clause_num;
      }
    } else if (free_var[thread_num] == 0) {
      /* Previous clauses is SAT, so feel free to overwrite them. */
      clause_candidate[thread_num] = clause_num;
      literal_candidate[thread_num] = best_literal;
      free_var[thread_num] = num_free;
    } else if (num_free > 0 && (num_free < free_var[thread_num] || 
           (num_free == free_var[thread_num] && abs(best_literal) < abs(literal_candidate[thread_num])) ||
           (num_free == free_var[thread_num] && abs(best_literal) == abs(literal_candidate[thread_num]) && clause_num < clause_candidate[thread_num]))) {
          /* Undecided clause that we prefer over previous undecided clauses. */
      clause_candidate[thread_num] = clause_num;
      literal_candidate[thread_num] = best_literal;
      free_var[thread_num] = num_free;
      /* If current clause is satisfying, then we don't care about it */
    }
    
    clause_num += stride;
  } 
  __syncthreads();
  if(thread_num<2){
  
  }
  /* Reduce arrays now */
  int sz = blockDim.x;
  while (sz > 1) {
    /* Proceed until we have one element left. Same reduction rules as above. */
    int i = thread_num;
    int j = i + sz / 2;
    if (thread_num < sz / 2) {
      if (free_var[j] == -1 || free_var[i] == -1) {
        /* If one of the two are NSAT */
        if ((free_var[j] == -1 && free_var[i] != -1) ||
            free_var[j] == -1 && clause_candidate[j] < clause_candidate[i]) {
            /* If j is NSAT and i is maybe NSAT or if j is NSAT and comes in an earlier clause */
            free_var[i] = free_var[j];
            clause_candidate[i] = clause_candidate[j];
        }
      } else if (free_var[i] == 0) {
          /* If i is maybe SAT */
          clause_candidate[i] = clause_candidate[j];
          literal_candidate[i] = literal_candidate[j];
          free_var[i] = free_var[j];
      } else if (free_var[j] > 0 && (free_var[j] < free_var[i] ||
           (free_var[i] == free_var[j] && abs(literal_candidate[j]) < abs(literal_candidate[i])) ||
           (free_var[i] == free_var[j] && abs(literal_candidate[j]) == abs(literal_candidate[i]) && clause_candidate[j] < clause_candidate[i]))) {
        clause_candidate[i] = clause_candidate[j];
        literal_candidate[i] = literal_candidate[j];
        free_var[i] = free_var[j];
      }
    }
    __syncthreads();
    sz /= 2;
  }
  if (thread_num == 0) {
    dev_clause_cand[blockIdx.x] = clause_candidate[0];
    dev_lit_cand[blockIdx.x] = literal_candidate[0];
    dev_free_var[blockIdx.x] = free_var[0];
  }
}


/*
	This function will find out if the current assignment will make the formula satifiable or unsatiable,
	if a conclusion is not yet reached, it will choose the clause with the least number of unassigned literals.  
*/
__host__ void mask_prop(int*best_clause, int*best_literal, int* num_free_literals){
	do
	{ 
    cudaMemcpy( assignment_device, assignment_host, (total_variables+1) * sizeof(int), cudaMemcpyHostToDevice );

		mask_prop_gpu<<<BLOCKS, THREADS>>>( num_free_literals_device,clause_candidates_device,literal_candidates_device, formula_device, assignment_device, clause_pointer_device,total_clauses, total_literals, total_variables);
    //read out the result
		cudaMemcpy( clause_candidates_host, clause_candidates_device, (BLOCKS)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy( literal_candidates_host, literal_candidates_device, (BLOCKS)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy( num_free_literals_host, num_free_literals_device, (BLOCKS)*sizeof(int), cudaMemcpyDeviceToHost);

		*num_free_literals=0; // if num_free_literals is -1, current assignment is unsatisfiable; 0 means solution is found; n > 0 means there are n free variables in the best_clause;        
		*best_clause=0;                         
		*best_literal=0;
		static int printlimit=0;
		// merge result from all block into a final value
		for (int i=0;i<BLOCKS;i++) 
		{                                
      if (
					( num_free_literals_host[i]==-1 && *num_free_literals!=-1) || 
					( num_free_literals_host[i]==-1 && *num_free_literals==-1 && *best_clause > clause_candidates_host[i]) || 
					*num_free_literals == 0 || 	
					(
					*num_free_literals > 0 && num_free_literals_host[i]>0 && (
															num_free_literals_host[i] < *num_free_literals || 
															(num_free_literals_host[i] == *num_free_literals && abs(literal_candidates_host[i])<abs(*best_literal)) || 
															(num_free_literals_host[i] == *num_free_literals && abs(literal_candidates_host[i])==abs(*best_literal) && clause_candidates_host[i]<*best_clause)
													    )
					)
				)
			{
        *num_free_literals=num_free_literals_host[i];
				*best_clause=clause_candidates_host[i];
				*best_literal=literal_candidates_host[i];
			}
      //printf("[%d]num: %d clause: %d iter %d \n", i, num_free_literals_host[i], clause_candidates_host[i], literal_candidates_host[i] );
		}
    


		if (*num_free_literals==1){
			int var = abs(*best_literal);
			assignment_host[var] = *best_literal > 0;
			level[var] = current_level;
			refs[var] = *best_clause;
			assignment_stack[assignment_stack_pointer] = *best_literal;
			assignment_stack_pointer++;
      //printf("sp: %d\n",  assignment_stack_pointer);
  //printf("stack: ");
  //for(int i =1 ; i< assignment_stack_pointer; i++)
  //  printf("%d ", assignment_stack[i]);
  //printf("\n");		  
		}	
	}
	while (*num_free_literals==1); // keep doing unit propogation until you can't
}


__host__ int backtrack(int level_to_jump){
  //printf("backtrack!!\n");
  //printf("before sp: %d\n",  assignment_stack_pointer);
  //printf("stack: ");
  //for(int i =1 ; i< assignment_stack_pointer; i++)
  //  printf("%d ", assignment_stack[i]);
  //printf("\n");
	int loop=assignment_stack_pointer>0;
	while (loop){ // this fixes trails and variables. level fixed on recursion
		int VARP=abs(assignment_stack[assignment_stack_pointer-1]);
		loop= assignment_stack_pointer>0 && ((level[VARP]>level_to_jump)   ||  (refs[VARP]>=0)  || (level[VARP]<=level_to_jump && refs[VARP]==-2));
		if (level[VARP]>level_to_jump || refs[VARP]>=0 || refs[VARP]==-2)
		{
		//printf("back var %d lev %d\n",VARP,level[VARP]);
		assignment_host[VARP] = -1;		// restore "unknown" status
		level[VARP] = -1;
		refs[VARP] = -1;
		assignment_stack_pointer--;
	    }
	    else
	    {
	    	if (level[VARP]<=level_to_jump && refs[VARP]==-1)
	    	{ // variable already tested -> I put the opposite case
	        	//printf("switch var %d at lev %d -> %d\n",VARP,level[VARP],-trail[trail[0]-1]);
	        	refs[VARP] = -2;
	        	assignment_host[VARP] = 1-assignment_host[VARP]; // inverse status      
	        	assignment_stack[assignment_stack_pointer-1]=-assignment_stack[assignment_stack_pointer-1]; // invert value on trail
	        	current_level=level[VARP];
	      	}	    
	    }
 	}
  //printf("aftter sp: %d\n",  assignment_stack_pointer);
  //printf("stack: ");
  //for(int i =1 ; i< assignment_stack_pointer; i++)
  //  printf("%d ", assignment_stack[i]);
  //printf("\n");
  return 0;

}

__host__ int twolevel_DPLL(){
  int num_free_literals=0; // if num_free_literals is -1, current assignment is unsatisfiable; 0 means solution is found; n > 0 means there are n free variables in the best_clause;        
  int best_clause=0;                         
  int best_literal=0;

  int var;
	short good=0;

	do 
  {    
    	// mask_propagation
    	mask_prop(&best_clause, &best_literal, &num_free_literals);
    	if (current_level<=0 && num_free_literals==-1){
      		return 0;
    	}
    	//************* UNSATISFIABLE ASSIGNMENT
    	if (num_free_literals < 0){ // At least one clause is false
      	//printf("TEST 103000\n");
      	good = 0;
      	refs[0] = best_clause;	// conflict clause
      	backtrack(current_level); //if i don't learn, i open brother (i keep current level)
    }  
    else if (num_free_literals == 0){  //************* FOUND A SOLUTION
      good = 1;  
    }      
    else if (num_free_literals > 0)  
    { 
  		
      var = abs(best_literal);
  		current_level++;
  		level[var] = current_level;
  		assignment_host[var] = best_literal > 0;
      //printf("assigning var %d to be %d\n", var, assignment_host[var]);

      //printf("sp: %d\n",  assignment_stack_pointer);
      //printf("stack: ");
      //for(int i =1 ; i< assignment_stack_pointer; i++)
      //  printf("%d ", assignment_stack[i]);
      //printf("\n");
  		assignment_stack[assignment_stack_pointer] = best_literal;

      assignment_stack_pointer++;
    }
    else{
      printf("something is wrong is twolevel_DPLL!!\n");
    }
  }
  while(current_level>0 && !good);
  return good;

}

__host__ int allocate_memory1(){
	cudaHostAlloc( (void**)&formula_host,		MAX_LITERALS  * sizeof(int), cudaHostAllocDefault );
  cudaMalloc((void**)&formula_device,         MAX_LITERALS * sizeof(int));
  cudaHostAlloc( (void**)&clause_pointer_host,MAX_CLAUSES * sizeof(int), cudaHostAllocDefault );
  cudaMalloc((void**)&clause_pointer_device,  MAX_CLAUSES * sizeof(int));
  return 0;		
}

__host__ int allocate_memory2(){
  //determine grid dimension
  cudaDeviceProp gpu_property;
  int gpu_device_handler;
  cudaGetDevice( &gpu_device_handler );
  cudaGetDeviceProperties( &gpu_property, gpu_device_handler );
  BLOCKS = MIN( ( total_clauses + THREADS - 1) / THREADS, 2*gpu_property.multiProcessorCount );

  cudaHostAlloc( (void**)&assignment_host,	(total_variables+1) * sizeof(int), cudaHostAllocDefault );
  cudaMalloc((void**)&assignment_device,  	(total_variables+1) * sizeof(int));

  //for prog_mask
  cudaHostAlloc( (void**)&num_free_literals_host,	(BLOCKS)*sizeof(int) , cudaHostAllocDefault );
  cudaMalloc((void**)&num_free_literals_device,  	(BLOCKS)*sizeof(int));
  
  cudaHostAlloc( (void**)&clause_candidates_host,	(BLOCKS)*sizeof(int) , cudaHostAllocDefault );
  cudaMalloc((void**)&clause_candidates_device,  	(BLOCKS)*sizeof(int));
  
  cudaHostAlloc( (void**)&literal_candidates_host,	(BLOCKS)*sizeof(int) , cudaHostAllocDefault );
  cudaMalloc((void**)&literal_candidates_device,  	(BLOCKS)*sizeof(int));

  //for backtracking
  level  = (int*) malloc(1000 * sizeof(int));
  refs   = (int*) malloc(1000 * sizeof(int));
  assignment_stack  = (int*) malloc(1000 * sizeof(int));


  return 0;
}


__host__ int initialize(){
  

  //initialize formula, clause pointer & assignment
	cudaMemcpy( formula_device, formula_host,    total_literals * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( clause_pointer_device, clause_pointer_host, (total_clauses+1) * sizeof(int), cudaMemcpyHostToDevice );
  for(int i =1 ; i < total_variables+1 ; i++ )
  {
    assignment_host[i]=-1;
  }

  //initialize back tracking
  for(int i=0;i<total_variables+1;i++)
  {
    level[i] = -1;
    refs[i] = -1;
  }
  current_level=0;
  assignment_stack_pointer=1;  


	return 0;
}


__host__ int deallocate_memory(){
	return 0;
}


__host__ int  main(int argc, char** argv) {
	cudaEvent_t start, stop;
	float elapsed_time_ms;
	
	if(argc==1)
		printf("usage: %s filename\n", argv[0] );

	allocate_memory1();//allocate memory that load_dimacs_cnf will require
	
	load_dimacs_cnf(argv[argc-1], &total_variables, &total_clauses, &total_literals);
	
 
  


	allocate_memory2();//allocate memory that its size depends on the input file 

	initialize();
  printf("formula:\n");
  for(int i =0 ; i < total_literals;i++){
    printf("%d ", formula_host[i]);
  }
  printf("\n");
  printf("clause pointer:\n");
  for(int i =0 ; i < total_clauses; i++){
    printf("%d ", clause_pointer_host[i]);
  }
  printf("\n");
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);
  cudaEventSynchronize(start);

	int returnval= twolevel_DPLL();
  cudaEventRecord(stop, 0);    
  cudaEventSynchronize(stop);    
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);  

  if(returnval){
    printf("the formula is satifiable\n");
    //for(int i =1 ; i< assignment_stack_pointer; i++)
    //  printf("%d\n", assignment_stack[i]);
  }
  else
    printf("the formula is unsatisfiable\n");


  
  printf("took %f seconds\n", elapsed_time_ms/1000);

	//deallocate_memory();//does it really matter tho? ;-) 
	return 0;
}
