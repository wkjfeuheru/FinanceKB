# Caddy 官方镜像不含限流插件，构建期用 xcaddy 打进去。
FROM caddy:2.8.4-builder AS builder

RUN xcaddy build --with github.com/mholt/caddy-ratelimit

FROM caddy:2.8.4

COPY --from=builder /usr/bin/caddy /usr/bin/caddy
