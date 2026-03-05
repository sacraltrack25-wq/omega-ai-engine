# OMEGA Unified вЂ” Node AI Engine + Python Harvesters + Encoder
# One container: Node on PORT, Python on 8000 (internal)

FROM node:20-alpine AS node-builder
WORKDIR /app/core
COPY core/package.json core/tsconfig.json ./
COPY core/src ./src
RUN npm install && npm run build

# Base: Node (has node) + Python + ffmpeg
FROM node:20-bookworm-slim
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -sf python3 /usr/bin/python && ln -sf pip3 /usr/bin/pip

WORKDIR /app
COPY harvesters/requirements.txt ./harvesters/
RUN pip3 install --no-cache-dir -r harvesters/requirements.txt
COPY harvesters ./harvesters

COPY --from=node-builder /app/core/dist ./core/dist
COPY --from=node-builder /app/core/node_modules ./core/node_modules
COPY core/package.json ./core/

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENV PORT=4000
EXPOSE 4000 8000
CMD ["/start.sh"]
