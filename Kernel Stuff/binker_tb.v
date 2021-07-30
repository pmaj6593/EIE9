module binker_tb();

	reg[2:0] im1, im2, im3, k1, k2, k3;
	wire[3:0] result;
	
	bin_ker multi(.im1(im1), .im2(im2), .im3(im3), .k1(k1), .k2(k2), .k3(k3), .result(result));
	
	initial begin
		im1 = 3'b101;
		im2 = 3'b110;
		im3 = 3'b011;
		k1 = 3'b111;
		k2 = 3'b111;
		k3 = 3'b111;
	end
	
endmodule 