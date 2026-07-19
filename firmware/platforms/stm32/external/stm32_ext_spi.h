#pragma once

#include "external.h"
#include "stm32g4xx_hal.h"

/*
 * Hardware configuration for the STM32 SPI slave external-controller transport.
 * All peripheral and GPIO assignments are caller-supplied; this module is
 * board-agnostic.
 *
 * Assumption: SCK, MISO, MOSI share a single GPIO port.
 * The SPI handle Init fields (mode, data size, polarity, etc.) are set by
 * nc_ext_spi_hw_init() to match the fixed protocol; the caller only supplies
 * the peripheral instance via hspi->Instance.
 */
typedef struct {
    SPI_HandleTypeDef *hspi;       /* SPI handle; hspi->Instance must be set */
    GPIO_TypeDef      *spi_port;   /* GPIO port for SCK, MISO, MOSI          */
    uint16_t           sck_pin;
    uint16_t           miso_pin;
    uint16_t           mosi_pin;
    uint8_t            spi_af;     /* GPIO alternate function number          */
    GPIO_TypeDef      *drdy_port;
    uint16_t           drdy_pin;
    uint32_t           timeout_ms;
} nc_ext_spi_config_t;

/*
 * Initialise GPIO clocks, SPI GPIO pins, DRDY output, and the SPI peripheral.
 * Call once before nc_ext_spi_make().
 */
void nc_ext_spi_hw_init(const nc_ext_spi_config_t *cfg);

/*
 * Build an nc_external_ctrl_t wired to the SPI slave transport.
 * cfg must remain valid for the lifetime of the controller.
 */
nc_external_ctrl_t nc_ext_spi_make(const nc_ext_spi_config_t *cfg);
