// Simple benchmark designs for DRiLLS reproduction
// Multiplier circuit - used in the original paper

module multiplier(a, b, product);
  input [7:0] a, b;
  output [15:0] product;
  assign product = a * b;
endmodule
