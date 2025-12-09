# Development Environment - Trading Bot Infrastructure

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "trading-bot-terraform-state"
    key            = "dev/terraform.tfstate"
    region         = "ap-southeast-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = "dev"
      ManagedBy   = "terraform"
    }
  }
}

variable "project_name" {
  default = "trading-bot"
}

variable "aws_region" {
  default = "ap-southeast-1"
}

variable "availability_zones" {
  default = ["ap-southeast-1a", "ap-southeast-1b"]
}

# VPC
module "vpc" {
  source = "../../modules/vpc"

  project_name       = var.project_name
  environment        = "dev"
  vpc_cidr           = "10.1.0.0/16"
  availability_zones = var.availability_zones
  private_subnet_cidrs = ["10.1.1.0/24", "10.1.2.0/24"]
  public_subnet_cidrs  = ["10.1.101.0/24", "10.1.102.0/24"]
}

# EKS - smaller for dev
module "eks" {
  source = "../../modules/eks"

  project_name        = var.project_name
  environment         = "dev"
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  kubernetes_version  = "1.28"
  node_instance_types = ["t3.medium"]
  node_desired_size   = 2
  node_min_size       = 1
  node_max_size       = 3
}

# RDS - minimal for dev
module "rds" {
  source = "../../modules/rds"

  project_name            = var.project_name
  environment             = "dev"
  vpc_id                  = module.vpc.vpc_id
  private_subnet_ids      = module.vpc.private_subnet_ids
  allowed_security_groups = [module.eks.cluster_security_group_id]
  instance_class          = "db.t3.micro"
  allocated_storage       = 20
  multi_az                = false
  deletion_protection     = false
  backup_retention_period = 1
}

# Redis - minimal for dev
module "elasticache" {
  source = "../../modules/elasticache"

  project_name            = var.project_name
  environment             = "dev"
  vpc_id                  = module.vpc.vpc_id
  private_subnet_ids      = module.vpc.private_subnet_ids
  allowed_security_groups = [module.eks.cluster_security_group_id]
  node_type               = "cache.t3.micro"
  num_cache_nodes         = 1
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "rds_endpoint" {
  value     = module.rds.endpoint
  sensitive = true
}

output "redis_endpoint" {
  value = module.elasticache.primary_endpoint
}
