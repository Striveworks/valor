FROM node:alpine

WORKDIR /web

COPY package.json .

COPY .npmrc .

RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]