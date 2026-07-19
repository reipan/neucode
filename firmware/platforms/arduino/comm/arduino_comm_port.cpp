/**
 * UART comm port - STM32Duino platform.
 * Routes to Arduino Serial (ST-Link VCP); comm.c / cmd_handler.c are platform-independent.
 */

#include <Arduino.h>

extern "C" {
#include "comm.h"
#include "cmd_handler.h"
}

#define RX_LINE_BUF_SIZE 128

static char rx_line_buf[RX_LINE_BUF_SIZE];
static size_t rx_line_len = 0;


static void arduino_comm_tx(const uint8_t *data, size_t len) {
    Serial.write(data, len);
}


static void arduino_comm_cmd(const char *cmd, size_t len) {
    nc_comm_cmd_handle_line(cmd, len);
}


void arduino_comm_init(void) {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}   // wait up to 3 s for USB CDC
    nc_comm_init(arduino_comm_tx, arduino_comm_cmd);
}

/**
 * Poll Serial for incoming bytes and forward to comm layer.
 * Call this once per main loop iteration.
 */
void arduino_comm_poll(void) {
    while (Serial.available()) {
        char c = (char)Serial.read();
        nc_comm_rx_bytes((const uint8_t *)&c, 1);

        // Simple line accumulator so we can echo for debug
        if (c == '\n' || c == '\r') {
            rx_line_len = 0;
        } else if (rx_line_len < RX_LINE_BUF_SIZE - 1) {
            rx_line_buf[rx_line_len++] = c;
            rx_line_buf[rx_line_len]   = '\0';
        }
    }
}
