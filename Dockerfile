FROM pufferai/puffer-deps:latest



RUN mkdir -p /puffertank
WORKDIR /puffertank

RUN git clone https://github.com/pufferai/pufferlib && pip3 install --user -e pufferlib/[cleanrl,atari]
RUN git clone --single-branch --depth=1 https://github.com/carperai/nmmo-environment && pip3 install --user -e nmmo-environment/[all]
RUN git clone --depth=1 https://github.com/carperai/nmmo-baselines

