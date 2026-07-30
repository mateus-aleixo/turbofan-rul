variable "project" {
  description = "Name prefix for every resource"
  type        = string
  default     = "conformal-rul"
}

variable "region" {
  description = "AWS region (eu-west-1 keeps latency low from Portugal)"
  type        = string
  default     = "eu-west-1"
}

variable "github_repo" {
  description = "GitHub repository allowed to deploy via OIDC, as owner/name"
  type        = string
  default     = "mateus-aleixo/conformal-rul"
}

variable "image_tag" {
  description = "Image tag the Lambda points at; CI moves the function to new tags"
  type        = string
  default     = "latest"
}

variable "budget_email" {
  description = "Email for the monthly cost-budget alarm"
  type        = string
}
