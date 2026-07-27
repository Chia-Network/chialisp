all:
	cargo build --release --no-default-features
	cargo build --release --target wasm32-unknown-unknown
	export RUSTFLAGS='--cfg=getrandom_backend="unsupported" -Ctarget-cpu=mvp'
	wasm-pack build
	npm link ./pkg
	(cd mock-test && npm link chialisp)
