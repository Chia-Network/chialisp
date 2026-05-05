#include <stdint.h>

struct env {
  uint32_t struct_type;
  void* env_ptr;
  void* stack_top;
  void** alloc_ptr;
};

uint32_t dispatch_code(uint32_t env, void* code) {
  return 0;
}
