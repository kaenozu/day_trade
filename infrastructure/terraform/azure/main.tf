# Issue #800 Phase 3: Azure環境 Terraform設定
# Container Instances + Azure SQL + Redis Cache構成

terraform {
  required_version = ">= 1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "day-trade-terraform-state"
    storage_account_name = "daytradetfstate"
    container_name       = "tfstate"
    key                  = "production/terraform.tfstate"
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }

    key_vault {
      purge_soft_delete_on_destroy = true
    }
  }
}

# データソース
data "azurerm_client_config" "current" {}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "day-trade-${var.environment}"
  location = var.azure_region

  tags = {
    Environment = var.environment
    Project     = "day-trade"
    ManagedBy   = "terraform"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "day-trade-${var.environment}-vnet"
  address_space       = [var.vnet_cidr]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = {
    Name = "day-trade-${var.environment}-vnet"
  }
}

# Subnets
resource "azurerm_subnet" "container_instances" {
  name                 = "container-instances"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.container_subnet_cidr]

  delegation {
    name = "delegation"

    service_delegation {
      name    = "Microsoft.ContainerInstance/containerGroups"
      actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
    }
  }
}

resource "azurerm_subnet" "database" {
  name                 = "database"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.database_subnet_cidr]

  service_endpoints = ["Microsoft.Sql"]
}

resource "azurerm_subnet" "cache" {
  name                 = "cache"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.cache_subnet_cidr]
}

# Network Security Groups
resource "azurerm_network_security_group" "container_instances" {
  name                = "day-trade-${var.environment}-container-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowHTTP"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_ranges    = ["8000", "8001", "8002"]
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = {
    Name = "day-trade-${var.environment}-container-nsg"
  }
}

resource "azurerm_network_security_group" "database" {
  name                = "day-trade-${var.environment}-database-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowSQL"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "1433"
    source_address_prefix      = var.container_subnet_cidr
    destination_address_prefix = "*"
  }

  tags = {
    Name = "day-trade-${var.environment}-database-nsg"
  }
}

# Associate NSG to Subnets
resource "azurerm_subnet_network_security_group_association" "container_instances" {
  subnet_id                 = azurerm_subnet.container_instances.id
  network_security_group_id = azurerm_network_security_group.container_instances.id
}

resource "azurerm_subnet_network_security_group_association" "database" {
  subnet_id                 = azurerm_subnet.database.id
  network_security_group_id = azurerm_network_security_group.database.id
}

# Azure SQL Server
resource "azurerm_mssql_server" "main" {
  name                         = "day-trade-${var.environment}-sql"
  resource_group_name          = azurerm_resource_group.main.name
  location                     = azurerm_resource_group.main.location
  version                      = "12.0"
  administrator_login          = var.sql_admin_username
  administrator_login_password = var.sql_admin_password

  minimum_tls_version = "1.2"

  azuread_administrator {
    login_username = var.sql_azuread_admin_login
    object_id      = data.azurerm_client_config.current.object_id
  }

  tags = {
    Name = "day-trade-${var.environment}-sql"
  }
}

# Azure SQL Database
resource "azurerm_mssql_database" "main" {
  name      = "day-trade-${var.environment}"
  server_id = azurerm_mssql_server.main.id

  sku_name                    = var.sql_database_sku
  max_size_gb                 = var.sql_database_max_size_gb
  auto_pause_delay_in_minutes = var.environment == "production" ? -1 : 60

  tags = {
    Name = "day-trade-${var.environment}-database"
  }
}

# SQL Server Firewall Rules
resource "azurerm_mssql_firewall_rule" "container_subnet" {
  name             = "ContainerSubnet"
  server_id        = azurerm_mssql_server.main.id
  start_ip_address = cidrhost(var.container_subnet_cidr, 0)
  end_ip_address   = cidrhost(var.container_subnet_cidr, -1)
}

# Azure Cache for Redis
resource "azurerm_redis_cache" "main" {
  name                = "day-trade-${var.environment}-redis"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku_name

  enable_non_ssl_port = false
  minimum_tls_version = "1.2"

  subnet_id = azurerm_subnet.cache.id

  redis_configuration {
    enable_authentication = true
  }

  tags = {
    Name = "day-trade-${var.environment}-redis"
  }
}

# Storage Account for Logs and Data
resource "azurerm_storage_account" "main" {
  name                     = "daytradestore${var.environment}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = var.environment == "production" ? "GRS" : "LRS"

  min_tls_version = "TLS1_2"

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = var.environment == "production" ? 30 : 7
    }
  }

  tags = {
    Name = "day-trade-${var.environment}-storage"
  }
}

# Storage Containers
resource "azurerm_storage_container" "ml_models" {
  name                  = "ml-models"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "data_cache" {
  name                  = "data-cache"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "backups" {
  name                  = "backups"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "logs" {
  name                  = "logs"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# Key Vault
resource "azurerm_key_vault" "main" {
  name                = "day-trade-${var.environment}-kv"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  soft_delete_retention_days = 90
  purge_protection_enabled   = var.environment == "production" ? true : false

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = [
      "Get", "List", "Update", "Create", "Import", "Delete", "Recover", "Backup", "Restore"
    ]

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Recover", "Backup", "Restore"
    ]

    certificate_permissions = [
      "Get", "List", "Update", "Create", "Import", "Delete", "Recover", "Backup", "Restore"
    ]
  }

  tags = {
    Name = "day-trade-${var.environment}-keyvault"
  }
}

# Key Vault Secrets
resource "azurerm_key_vault_secret" "sql_connection_string" {
  name         = "sql-connection-string"
  value        = "Server=tcp:${azurerm_mssql_server.main.fully_qualified_domain_name},1433;Initial Catalog=${azurerm_mssql_database.main.name};Persist Security Info=False;User ID=${var.sql_admin_username};Password=${var.sql_admin_password};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"
  key_vault_id = azurerm_key_vault.main.id

  tags = {
    Name = "day-trade-${var.environment}-sql-connection"
  }
}

resource "azurerm_key_vault_secret" "redis_connection_string" {
  name         = "redis-connection-string"
  value        = "${azurerm_redis_cache.main.hostname}:${azurerm_redis_cache.main.ssl_port},password=${azurerm_redis_cache.main.primary_access_key},ssl=True,abortConnect=False"
  key_vault_id = azurerm_key_vault.main.id

  tags = {
    Name = "day-trade-${var.environment}-redis-connection"
  }
}

resource "azurerm_key_vault_secret" "storage_connection_string" {
  name         = "storage-connection-string"
  value        = azurerm_storage_account.main.primary_connection_string
  key_vault_id = azurerm_key_vault.main.id

  tags = {
    Name = "day-trade-${var.environment}-storage-connection"
  }
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "day-trade-${var.environment}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = var.environment == "production" ? 90 : 30

  tags = {
    Name = "day-trade-${var.environment}-logs"
  }
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "day-trade-${var.environment}-insights"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.main.id

  tags = {
    Name = "day-trade-${var.environment}-insights"
  }
}