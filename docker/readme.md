# Docker Usage Instructions

## Prerequisites
- Ensure you have Docker installed on your machine. You can download it from [Docker's official website](https://www.docker.com/get-started).

## Building the Docker Image
To build the Docker image, navigate to the directory containing the Dockerfile and run the following command:

```bash
docker build -t <image_name> .
```

Replace `<image_name>` with your desired image name.

## Running the Docker Container
After building the image, you can run a container using:

```bash
docker run -it <image_name>
```

## Additional Commands
- To list all Docker images:
  ```bash
  docker images
  ```

- To stop a running container:
  ```bash
  docker stop <container_id>
  ```

- To remove a container:
  ```bash
  docker rm <container_id>
  ```

## Notes
- Ensure that any necessary ports are exposed in the Dockerfile for proper access.
- Check the specific Dockerfile for any additional environment variables or configurations required.
