all:
	cargo build --release --no-default-features
	cargo build --release --target wasm32-unknown-unknown
	export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS='--cfg=getrandom_backend="unsupported" -Ctarget-cpu=mvp -Ctarget-feature=+reference-types' && wasm-pack build
	npm link ./pkg
	(cd mock-test && npm link chialisp)
