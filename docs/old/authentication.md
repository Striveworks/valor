# Authentication

The API can be run without authentication (by default) or with authentication provided by [auth0](https://auth0.com/). A small react app (code at `web/`)

## Backend

To enable authentication for the backend either set the environment variables `AUTH_DOMAIN`, `AUTH_AUDIENCE`, and `AUTH_ALGORITHMS` or put them in a file named `.env.auth` in the `api` directory. An example of such a file is

```
AUTH0_DOMAIN="velour.us.auth0.com"
AUTH0_AUDIENCE="https://velour.striveworks.us/"
AUTH0_ALGORITHMS="RS256"
```

## Frontend

For the web UI either set the environment variables `VITE_AUTH0_DOMAIN`, `VITE_AUTH0_CLIENT_ID`, `VITE_AUTH0_CALLBACK_URL` and `VITE_AUTH0_AUDIENCE` or put them in a file named `.env` in the `web` directory. The URL to the should also be provided under the `VITE_BACKEND_URL`.  An example of such a file is:

```
VITE_AUTH0_DOMAIN=velour.us.auth0.com
VITE_AUTH0_CLIENT_ID=<AUTH0 CLIENT ID>
VITE_AUTH0_CALLBACK_URL=http://localhost:3000/callback
VITE_AUTH0_AUDIENCE=https://velour.striveworks.us/
VITE_BACKEND_URL=http://localhost:8000
```

Currently, the Striveworks UI Library, Minerva, is closed-source.  This requires access to the Striveworks Repo and an access token.
In order start the web container for Velour, a `.npmrc` file is required in the `web` directory.  Replace `GITHUB AUTH TOKEN` in the following example file:

```
@striveworks:registry = "https://npm.pkg.github.com/"
//npm.pkg.github.com/:_authToken = <GITHUB AUTH TOKEN>
```

## Testing auth

All tests mentioned above run without authentication except for `integration_tests/test_client_auth.py`. Running this test requires setting the envionment variables `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_CLIENT_ID`, and `AUTH0_CLIENT_SECRET` accordingly.

# Deployment settings

For deploying behind a proxy or with external routing, the environment variable `API_ROOT_PATH` can be set in the backend, which sets the `root_path` arguement to `fastapi.FastAPI` (see https://fastapi.tiangolo.com/advanced/behind-a-proxy/#setting-the-root_path-in-the-fastapi-app)