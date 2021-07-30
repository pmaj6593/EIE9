module bin_ker(im1, im2, im3, k1, k2, k3, result);

	input [2:0] im1, im2, im3, k1, k2, k3; 
	output reg[3:0] result;
	
	wire r1, r2, r3, r4, r5, r6,r7, r8, r9;
	
	assign r1 = (im1[2] * k1[2]);
	assign r2 = (im1[1] * k1[1]);
	assign r3 = (im1[0] * k1[0]);
	assign r4 = (im2[2] * k2[2]);
	assign r5 = (im2[1] * k2[1]);
	assign r6 = (im2[0] * k2[0]);
	assign r7 = (im3[2] * k3[2]);
	assign r8 = (im3[1] * k3[1]);
	assign r9 = (im3[0] * k3[0]);
	
	initial begin
		#1
		result = 6'b000000;
		#1
		result = result + r1;
		#1
		result = result + r2;
		#1
		result = result + r3;
		#1
		result = result + r4;
		#1
		result = result + r5;
		#1
		result = result + r6;
		#1
		result = result + r7;
		#1
		result = result + r8;
		#1
		result = result + r9;
	end
endmodule 