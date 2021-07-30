`timescale 1 ps / 1 ps
module kernal33(im1, im2, im3, k1, k2, k3, result);

	input[47:0] im1, im2, im3, k1, k2, k3;
	output[31:0] result;
	
	reg clk = 1'b0;
	reg [47:0] IM_mask, kernel_mask;
	
	stuff_0002 mult(.clock0(clk), .dataa_0(IM_mask[47:32]), .datab_0(kernel_mask[47:32]), .result(result));
	
	initial begin
		clk = 1'b0;
		#1
		IM_mask = im1; kernel_mask = k1; clk = 1'b1;
		#1
		clk = 1'b0;
		#1
		IM_mask = im1 << 16; kernel_mask = k1 << 16; clk = 1'b1;
		#1
		clk = 1'b0;
		#1
		IM_mask = im1 << 32; kernel_mask = k1 << 32; clk = 1'b1;
		#1
		clk = 1'b0;
		#1
		IM_mask = im2; kernel_mask = k2; clk = 1'b1;
		#1
		clk = 1'b0;
		#1
		IM_mask = im2 << 16; kernel_mask = k2 << 16; clk = 1'b1;
		#1
		clk = 1'b0;
		#1
		IM_mask = im2 << 32; kernel_mask = k2 << 32; clk = 1'b1;
		#1
		clk = 1'b0;
		#1
		IM_mask = im3; kernel_mask = k3; clk = 1'b1;
		#1
		clk = 1'b0;
		#1
		IM_mask = im3 << 16; kernel_mask = k3 << 16; clk = 1'b1;
		#1
		clk = 1'b0;
		#1
		IM_mask = im3 << 32; kernel_mask = k3 << 32; clk = 1'b1;
	end

endmodule 