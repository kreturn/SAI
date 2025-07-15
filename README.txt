
1: I only used C++98 features since our system is based on Centos 6.8

2: Source code is in saidemo, should be able to compile in Centos 6.8
	cmake .
	make 

3: There two set of data in datademo: a line with 744 cdps and 1 cdp only
   Configure lineconfig.par or 1cdpconfig.par files before running 4

4: run `saidemo/bin/sai lineconfig.par` should start processing on a line
   with 744 cdps case. 1cdpconfig.par has only one cdp

5: Where GPU parallel is needed: 
	* the for loop for traces in src/main.cc, ecah call goes to:
		ModelSimulator pss;
	        pss.simulate( s_data, w_data, ang, bm, wd );

	* the above `simulate` has two key steps: (in src/pss.cc)
		1:  L1LS::solver call: returned for now since it's 
		    the key step to achive first
		2:  MyOpt mo; mo.executeConstrained(...) is the second key step

6: The OpenMP version works now, I started writing Cuda on L1LS algo, but am
   having some difficulties at the moment. 
	Thinking of using Cuda Thrust or cusp for Matrix and Vector handling,
        what would you recommand? or use raw pointers directly.

7: You may disable CMAKE_CXX_FLAGS -o3 option in CMakeLists.txt if debug

8: Edit par file based on your data directory



	

