// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "odin_aer_bridge.v" - the synthesizable AER-in driver for one ODIN core.
// Original work; it drives the vendored link and derives no text from it.
//
// CROSS-LANGUAGE CONTRACT (doc Sec.2.2, ChFrenkel/ODIN @ 1781931): the input
// link is a FOUR-PHASE handshake with a double-latching barrier on REQ, so an
// event is complete only after REQ has gone high, ACK has risen, REQ has gone
// low and ACK has fallen again. Events are issued ONE AT A TIME, which is what
// makes the exporter's canonical drain order the order the crossbar sees --
// the whole point of the per-event soma law.

`timescale 1ns/1ps

module odin_aer_bridge #(
    parameter ADDR_BITS = 17
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    input  wire [ADDR_BITS-1:0] addr,
    input  wire                 aer_ack,
    output reg                  aer_req,
    output reg  [ADDR_BITS-1:0] aer_addr,
    output reg                  busy,
    output reg                  timeout
);

    // The drain of one event is bounded by the core's own sweep (512 cycles
    // per event on the stock geometry) plus output-handshake stalls; this
    // guard is far above that bound and exists so a wedged link reports a
    // FAILURE instead of hanging the kernel forever.
    localparam [31:0] ACK_GUARD = 32'd2000000;

    reg [31:0] guard;
    reg [1:0]  state;

    localparam S_IDLE     = 2'd0;
    localparam S_WAIT_ACK = 2'd1;
    localparam S_DROP_REQ = 2'd2;

    always @(posedge clk) begin
        if (rst) begin
            aer_req  <= 1'b0;
            aer_addr <= {ADDR_BITS{1'b0}};
            busy     <= 1'b0;
            timeout  <= 1'b0;
            guard    <= 32'd0;
            state    <= S_IDLE;
        end else begin
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        aer_addr <= addr;
                        aer_req  <= 1'b1;
                        busy     <= 1'b1;
                        guard    <= 32'd0;
                        state    <= S_WAIT_ACK;
                    end
                end
                S_WAIT_ACK: begin
                    if (aer_ack) begin
                        aer_req <= 1'b0;
                        guard   <= 32'd0;
                        state   <= S_DROP_REQ;
                    end else if (guard == ACK_GUARD) begin
                        timeout <= 1'b1;
                        busy    <= 1'b0;
                        aer_req <= 1'b0;
                        state   <= S_IDLE;
                    end else begin
                        guard <= guard + 32'd1;
                    end
                end
                default: begin
                    if (!aer_ack) begin
                        busy  <= 1'b0;
                        state <= S_IDLE;
                    end else if (guard == ACK_GUARD) begin
                        timeout <= 1'b1;
                        busy    <= 1'b0;
                        state   <= S_IDLE;
                    end else begin
                        guard <= guard + 32'd1;
                    end
                end
            endcase
        end
    end

endmodule
