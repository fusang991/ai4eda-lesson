// 8-bit adder benchmark
module adder(a, b, sum, carry);
  input [7:0] a, b;
  output [7:0] sum;
  output carry;
  assign {carry, sum} = a + b;
endmodule
