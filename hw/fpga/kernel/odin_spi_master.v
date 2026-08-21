// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "odin_spi_master.v" - a synthesizable SPI master for one vendored ODIN core.
// Original work: it drives the vendored slave and derives no text from it.
//
// CROSS-LANGUAGE CONTRACT, transcribed from ChFrenkel/ODIN @ 1781931
// (src/spi_slave.v:82-146, doc Sec.2.1) and identical to the one the
// cosimulation testbench implements in `hw/tb/tb_odin_core.v`:
//   * one transaction is 40 SCK edges, 20-bit address then 20-bit data, MSB
//     first; MOSI is sampled on posedge SCK and MISO changes on negedge SCK;
//   * SCK must be at least 4x slower than CLK, so this master runs SCK = CLK/4
//     exactly (one bit = four CLK cycles, SCK high for the middle two);
//   * a read byte appears on MISO during the LAST EIGHT bit slots, so the
//     shift register below is read out as `rdata[7:0]` when the frame ends.

`timescale 1ns/1ps

module odin_spi_master (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [19:0] addr,
    input  wire [19:0] data,
    input  wire        miso,
    output reg         sck,
    output reg         mosi,
    output reg         busy,
    output reg  [19:0] rdata
);

    localparam FRAME_BITS = 6'd40;
    localparam TAIL_CYCLES = 3'd4;

    reg [39:0] frame;
    reg [5:0]  bit_index;
    reg [1:0]  phase;
    reg [2:0]  tail;
    reg [1:0]  state;

    localparam S_IDLE = 2'd0;
    localparam S_SHIFT = 2'd1;
    localparam S_TAIL = 2'd2;

    always @(posedge clk) begin
        if (rst) begin
            sck       <= 1'b0;
            mosi      <= 1'b0;
            busy      <= 1'b0;
            rdata     <= 20'd0;
            frame     <= 40'd0;
            bit_index <= 6'd0;
            phase     <= 2'd0;
            tail      <= 3'd0;
            state     <= S_IDLE;
        end else begin
            case (state)
                S_IDLE: begin
                    sck  <= 1'b0;
                    busy <= 1'b0;
                    if (start) begin
                        frame     <= {addr, data};
                        rdata     <= 20'd0;
                        bit_index <= 6'd0;
                        phase     <= 2'd0;
                        busy      <= 1'b1;
                        state     <= S_SHIFT;
                    end
                end
                S_SHIFT: begin
                    case (phase)
                        2'd0: begin
                            mosi  <= frame[39];
                            sck   <= 1'b0;
                            phase <= 2'd1;
                        end
                        2'd1: begin
                            sck   <= 1'b1;
                            phase <= 2'd2;
                        end
                        2'd2: begin
                            // MISO is stable across the SCK-high window; the
                            // slave changed it on the preceding negedge.
                            rdata <= {rdata[18:0], miso};
                            phase <= 2'd3;
                        end
                        default: begin
                            sck   <= 1'b0;
                            frame <= {frame[38:0], 1'b0};
                            phase <= 2'd0;
                            if (bit_index == FRAME_BITS - 6'd1) begin
                                tail  <= TAIL_CYCLES;
                                state <= S_TAIL;
                            end else begin
                                bit_index <= bit_index + 6'd1;
                            end
                        end
                    endcase
                end
                default: begin
                    sck <= 1'b0;
                    if (tail == 3'd0) begin
                        busy  <= 1'b0;
                        state <= S_IDLE;
                    end else begin
                        tail <= tail - 3'd1;
                    end
                end
            endcase
        end
    end

endmodule
