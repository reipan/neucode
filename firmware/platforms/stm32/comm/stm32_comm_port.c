#include "cmd_handler.h"
#include "main.h"
#include "comm.h"
#include "stm32g4xx_hal_uart.h"
#include <stdint.h>
#include <string.h>
#include "stm32_comm_port.h"

#define NC_COMM_TX_BUF_SIZE 128

extern UART_HandleTypeDef hlpuart1;
static uint8_t stm32_comm_port_rx_byte;

static uint8_t stm32_comm_port_tx_buf[NC_COMM_TX_BUF_SIZE];
static size_t stm32_comm_port_tx_buf_len = 0;
static volatile bool stm32_comm_port_tx_busy = false;

/**
 * Transmits buffer data over the STM32 communication port (non-blocking).
 *
 * @param data Pointer to the buffer containing the data to transmit.
 * @param len  Number of bytes to transmit from the buffer.
 */
static void stm32_comm_port_tx(const uint8_t *data, size_t len) {
    if (!data || len == 0 || stm32_comm_port_tx_busy) return;

    // truncate
    if (len > NC_COMM_TX_BUF_SIZE) {
        len = NC_COMM_TX_BUF_SIZE;
    }

    memcpy(stm32_comm_port_tx_buf, data, len);
    stm32_comm_port_tx_buf_len = len;
    stm32_comm_port_tx_busy = true;

    HAL_UART_Transmit_IT(&hlpuart1, stm32_comm_port_tx_buf, (uint16_t)stm32_comm_port_tx_buf_len);
}

/**
 * Handles incoming command data for the STM32 communication port.
 *
 * @param cmd Pointer to the command string to be handled.
 * @param len Length of the command string.
 */
static void stm32_comm_port_cmd_handler(const char *cmd, size_t len) {
    nc_comm_cmd_handle_line(cmd, len);
}

/**
 * Initializes the STM32 communication port.
 *
 * This function sets up the necessary hardware and configuration
 * for the communication port on STM32 platforms.
 */
void stm32_comm_port_init(void) {
    nc_comm_init(stm32_comm_port_tx, stm32_comm_port_cmd_handler);
    HAL_UART_Receive_IT(&hlpuart1, &stm32_comm_port_rx_byte, 1);
}

/**
 * UART receive complete callback.
 *
 * This function is called by the HAL library when a UART receive operation is completed.
 *
 * @param huart Pointer to the UART handle structure containing more information about the UART peripheral.
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == LPUART1) {
        nc_comm_rx_bytes(&stm32_comm_port_rx_byte, 1);
        HAL_UART_Receive_IT(&hlpuart1, &stm32_comm_port_rx_byte, 1);
    }
}

/**
 * UART transmit complete callback.
 *
 * This function is called by the HAL library when a UART transmit operation is completed.
 *
 * @param huart Pointer to the UART handle structure containing more information about the UART peripheral.
 */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == LPUART1) {
        stm32_comm_port_tx_busy = false;
    }
}