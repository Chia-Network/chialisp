FROM alpine:latest
WORKDIR /root
RUN apk add git curl python3 openssl openssl-dev perl linux-headers make gcc musl-dev patch patchelf py3-pip
RUN curl --proto '=https' --tlsv1.2 -sSf -o rustup https://sh.rustup.rs
RUN sh ./rustup -y
RUN sh -c "source ~/.cargo/env && rustup component add --toolchain 1.97.1-x86_64-unknown-linux-musl cargo"
WORKDIR /root/chialisp
COPY ./build-alpine.sh /root
