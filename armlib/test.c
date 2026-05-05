#include "sha256.h"

void _start() {
	char buf[10] = "hello";
	char hash[32];
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, buf, 10);
	sha256_final(&ctx, hash);
}
