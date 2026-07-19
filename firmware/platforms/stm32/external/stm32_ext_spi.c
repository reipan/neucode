#include "stm32_ext_spi.h"

#include <math.h>
#include <string.h>

/* Enable the RCC clock for the given GPIO port. */
static void enable_gpio_clk(GPIO_TypeDef *port) {
    if (port == GPIOA) { __HAL_RCC_GPIOA_CLK_ENABLE(); return; }
    if (port == GPIOB) { __HAL_RCC_GPIOB_CLK_ENABLE(); return; }
    if (port == GPIOC) { __HAL_RCC_GPIOC_CLK_ENABLE(); return; }
    if (port == GPIOD) { __HAL_RCC_GPIOD_CLK_ENABLE(); return; }
    if (port == GPIOE) { __HAL_RCC_GPIOE_CLK_ENABLE(); return; }
}

/* Enable the RCC clock for the SPI peripheral in hspi->Instance. */
static void enable_spi_clk(SPI_HandleTypeDef *hspi) {
    if (hspi->Instance == SPI1) { __HAL_RCC_SPI1_CLK_ENABLE(); return; }
    if (hspi->Instance == SPI2) { __HAL_RCC_SPI2_CLK_ENABLE(); return; }
    if (hspi->Instance == SPI3) { __HAL_RCC_SPI3_CLK_ENABLE(); return; }
}

void nc_ext_spi_hw_init(const nc_ext_spi_config_t *cfg) {
    enable_gpio_clk(cfg->spi_port);
    enable_gpio_clk(cfg->drdy_port);
    enable_spi_clk(cfg->hspi);

    GPIO_InitTypeDef g = {0};
    g.Pin       = cfg->sck_pin | cfg->miso_pin | cfg->mosi_pin;
    g.Mode      = GPIO_MODE_AF_PP;
    g.Pull      = GPIO_NOPULL;
    g.Speed     = GPIO_SPEED_FREQ_HIGH;
    g.Alternate = cfg->spi_af;
    HAL_GPIO_Init(cfg->spi_port, &g);

    g.Pin       = cfg->drdy_pin;
    g.Mode      = GPIO_MODE_OUTPUT_PP;
    g.Pull      = GPIO_NOPULL;
    g.Alternate = 0;
    HAL_GPIO_Init(cfg->drdy_port, &g);
    HAL_GPIO_WritePin(cfg->drdy_port, cfg->drdy_pin, GPIO_PIN_RESET);

    cfg->hspi->Init.Mode           = SPI_MODE_SLAVE;
    cfg->hspi->Init.Direction      = SPI_DIRECTION_2LINES;
    cfg->hspi->Init.DataSize       = SPI_DATASIZE_8BIT;
    cfg->hspi->Init.CLKPolarity    = SPI_POLARITY_LOW;
    cfg->hspi->Init.CLKPhase       = SPI_PHASE_1EDGE;
    cfg->hspi->Init.NSS            = SPI_NSS_SOFT;
    cfg->hspi->Init.FirstBit       = SPI_FIRSTBIT_MSB;
    cfg->hspi->Init.TIMode         = SPI_TIMODE_DISABLE;
    cfg->hspi->Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
    HAL_SPI_Init(cfg->hspi);
}

/*
 * TX (MISO): [sp float32 LE][y float32 LE]  -- STM32 -> master
 * RX (MOSI): [u  float32 LE][padding 4 B]   -- master -> STM32
 *
 * DRDY is asserted before HAL_SPI_TransmitReceive so the master can
 * detect the edge and start clocking within cfg->timeout_ms.
 * Flag-clear on timeout (~1 us) keeps recovery fast enough that the
 * master needs only ~50 us of settle before the next transfer.
 */
static float spi_exchange(float sp, float y, void *ctx) {
    nc_ext_spi_config_t *cfg = (nc_ext_spi_config_t *)ctx;

    uint8_t tx_buf[8] = {0};
    memcpy(&tx_buf[0], &sp, sizeof(float));
    memcpy(&tx_buf[4], &y,  sizeof(float));

    uint8_t rx_buf[8] = {0};

    HAL_GPIO_WritePin(cfg->drdy_port, cfg->drdy_pin, GPIO_PIN_SET);
    HAL_StatusTypeDef status = HAL_SPI_TransmitReceive(
        cfg->hspi, tx_buf, rx_buf, sizeof(rx_buf), cfg->timeout_ms
    );
    HAL_GPIO_WritePin(cfg->drdy_port, cfg->drdy_pin, GPIO_PIN_RESET);

    if (status != HAL_OK) {
        __HAL_SPI_CLEAR_OVRFLAG(cfg->hspi);
        __HAL_SPI_CLEAR_MODFFLAG(cfg->hspi);
        cfg->hspi->State = HAL_SPI_STATE_READY;
        return NAN;
    }

    float u;
    memcpy(&u, &rx_buf[0], sizeof(float));
    return u;
}

nc_external_ctrl_t nc_ext_spi_make(const nc_ext_spi_config_t *cfg) {
    nc_external_ctrl_t ext = {
        .exchange   = spi_exchange,
        .ctx        = (void *)cfg,
        .timeout_us = cfg->timeout_ms * 1000U,
    };
    return ext;
}
