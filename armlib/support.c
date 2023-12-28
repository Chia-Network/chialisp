#include <stdint.h>
#include "sha256.h"

void *memset(void *target, int val, int size) {
	char *ptr = target;
	while (size--) {
		*ptr++ = val;
	}
	return target;
}

int memcmp(void *target, void *src, int size) {
  unsigned char *ptr1 = target, *ptr2 = src;
  while (size--) {
    int p = *ptr1++ - *ptr2++;
    if (p) {
      return p;
    }
  }
  return 0;
}

void sha256tree(uint8_t *hash, uint32_t *code) {
  SHA256_CTX ctx;
  sha256_init(&ctx);
  if ((code[0] & 1) == 1) {
    sha256_update(&ctx, "\x01", 1);
    sha256_update(&ctx, ((uint8_t *)&code[1]), code[0] / 2);
  } else {
    uint8_t subhash[32];
    sha256_init(&ctx);
    sha256_update(&ctx, "\x02", 1);
    sha256tree(subhash, (uint32_t *)code[0]);
    sha256_update(&ctx, subhash, sizeof(subhash));
    sha256tree(subhash, (uint32_t *)code[1]);
    sha256_update(&ctx, subhash, sizeof(subhash));
  }
  sha256_final(&ctx, hash);
}
