target "build" {
  dockerfile = "Dockerfile"
  tags = ["proyecto-docker:bake"]
}

target "validate-build" {
  inherits = ["build"]
  call = "check"
}
