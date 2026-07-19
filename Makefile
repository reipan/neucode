CC			:= gcc
CFLAGS		:= -O2 -std=c11 -Wall -Wextra -MMD -MP
INCLUDES	:= -Ineucode/c_src/include
LDFLAGS		:= -lm
PIC_CFLAGS	:= -fPIC

BIN_DIR		:= bin
BUILD_DIR	:= build

UNITY_DIR		:= vendor/unity
UNITY_SRC		:= $(UNITY_DIR)/unity.c
TEST_INCLUDES	:= $(INCLUDES) -I$(UNITY_DIR)

# LIBRARY
LIB_NAME		:= neucode
LIB_TARGET_A	:= $(BIN_DIR)/lib$(LIB_NAME).a
LIB_TARGET_SO	:= $(BIN_DIR)/lib$(LIB_NAME).so

LIB_SRCS := $(wildcard neucode/c_src/src/*.c)

# Static library objects and dependencies (No changes needed here)
LIB_OBJS 	:= $(patsubst %.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))
LIB_DEPS 	:= $(LIB_OBJS:.o=.d)
# PIC objects for shared library (No changes needed here)
LIB_PIC_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.pic.o,$(LIB_SRCS))
LIB_PIC_DEPS := $(LIB_PIC_OBJS:.o=.d)

# BINARY TEST EXAMPLE (uses the library)
EXAMPLE_TARGET := $(BIN_DIR)/example
#! CHANGED: Assuming your example source file is now in a root-level examples dir
EXAMPLE_SRCS	:= examples/api_sim.c
EXAMPLE_OBJS	:= $(patsubst %.c,$(BUILD_DIR)/%.o,$(EXAMPLE_SRCS))
EXAMPLE_DEPS	:= $(EXAMPLE_OBJS:.o=.d)

# UNIT TESTS
TEST_TARGET := $(BIN_DIR)/test_runner
TEST_SRCS   := \
	$(UNITY_SRC) \
	$(LIB_SRCS) \
	tests/c_tests/test_runner.c \
	tests/c_tests/test_control_loops.c

TEST_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))
TEST_DEPS := $(TEST_OBJS:.o=.d)

.PHONY: all clean test run-tests example lib dirs
all: dirs $(LIB_TARGET_A) $(LIB_TARGET_SO) $(EXAMPLE_TARGET) $(TEST_TARGET)

dirs:
	mkdir -p $(BIN_DIR) $(BUILD_DIR)

# Library rules
lib: $(LIB_TARGET_A) $(LIB_TARGET_SO)

CC_MSG = "  CC   "
LD_MSG = "  LD   "
AR_MSG = "  AR   "
CLEAN_MSG = "  CLEAN"
RUN_MSG = "  RUN  "

$(LIB_TARGET_A): $(LIB_OBJS)
	@echo $(AR_MSG) $@
	@mkdir -p $(dir $@)
	ar rcs $@ $(LIB_OBJS)

$(LIB_TARGET_SO): $(LIB_PIC_OBJS)
	@echo $(LD_MSG) $@
	@mkdir -p $(dir $@)
	$(CC) -shared -o $@ $(LIB_PIC_OBJS) $(LDFLAGS)

# Test rules
test: $(TEST_TARGET)
	@echo $(RUN_MSG) $@
	@$(TEST_TARGET)

run-tests: test

$(TEST_TARGET): $(TEST_OBJS) | dirs
	@echo $(LD_MSG) $@
	$(CC) $(CFLAGS) $(TEST_INCLUDES) $(TEST_OBJS) $(LDFLAGS) -o $@

# Example rules
example: $(EXAMPLE_TARGET)

$(EXAMPLE_TARGET): $(EXAMPLE_OBJS) $(LIB_TARGET_A) | dirs
	@echo $(LD_MSG) $@
	$(CC) $(CFLAGS) $(INCLUDES) $(EXAMPLE_OBJS) $(LIB_TARGET_A) $(LDFLAGS) -o $@

# Rule for compiling library source files
$(BUILD_DIR)/neucode/c_src/src/%.o: neucode/c_src/src/%.c
	@echo $(CC_MSG) $@
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Rule for compiling library source files for the shared library (PIC)
$(BUILD_DIR)/neucode/c_src/src/%.pic.o: neucode/c_src/src/%.c
	@echo $(CC_MSG) $@
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PIC_CFLAGS) $(INCLUDES) -c $< -o $@

# Rule for compiling example source files
$(BUILD_DIR)/examples/%.o: examples/%.c
	@echo $(CC_MSG) $@
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Rule for compiling C test files
$(BUILD_DIR)/tests/c_tests/%.o: tests/c_tests/%.c
	@echo $(CC_MSG) $@
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(TEST_INCLUDES) -c $< -o $@

# Rule for compiling Unity
$(BUILD_DIR)/vendor/unity/%.o: vendor/unity/%.c
	@echo $(CC_MSG) $@
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(TEST_INCLUDES) -c $< -o $@

-include $(LIB_DEPS) $(LIB_PIC_DEPS) $(EXAMPLE_DEPS) $(TEST_DEPS)

# Clean
clean:
	@echo $(CLEAN_MSG)
	rm -rf $(BUILD_DIR) $(BIN_DIR)
