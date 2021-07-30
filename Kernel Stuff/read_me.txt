Binary Kernel:
- Uses AND gate for binary values 0 and 1 to get multiplication between 0 and 1
- Can also be used for binary -1 and 1, which is an XNOR gate of 0 and 1
- Testbench called: binker_tb,
	- Runs tests on a 3x3 binary input and kernal, and rows 1-3 are called im1-im3, and kernel rows 1 - 3 are called k1 - k3
- Code: Doing an AND statement with the different bytes and then adding them. 

Full Precision Kernel: 
- Uses multiply-accumulator called 'stuff0002' and maps to corresponding ports
- The final variable of result is sent in and preserved, then outputted. 
- Testbench called kernal33_tb and uses code kernal33. Note its a 3x3 kernal, which has 16 bits weights and returns 32 bit result
- Other stuff is just for IP cores stuff