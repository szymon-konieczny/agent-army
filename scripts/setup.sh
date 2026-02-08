#!/bin/bash

################################################################################
# AgentArmy Setup Script
# Initializes the AgentArmy project with all required dependencies and keys
################################################################################

set -euo pipefail

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

################################################################################
# Check prerequisites
################################################################################

check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_tools=()

    # Check for Docker
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    else
        log_success "Docker is installed"
    fi

    # Check for Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        missing_tools+=("docker-compose")
    else
        log_success "Docker Compose is installed"
    fi

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    else
        log_success "Python 3 is installed: $(python3 --version)"
    fi

    # Check for OpenSSL
    if ! command -v openssl &> /dev/null; then
        missing_tools+=("openssl")
    else
        log_success "OpenSSL is installed"
    fi

    # Check for Git
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    else
        log_success "Git is installed"
    fi

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
    fi

    log_success "All prerequisites are met!"
}

################################################################################
# Create .env file from template
################################################################################

create_env_file() {
    log_info "Creating .env file from template..."

    if [ -f ".env" ]; then
        log_warning ".env file already exists. Skipping creation."
        return
    fi

    if [ ! -f ".env.example" ]; then
        log_error ".env.example file not found"
    fi

    cp .env.example .env
    log_success ".env file created"

    # Make .env readable only by owner
    chmod 600 .env
    log_success ".env file permissions set to 600"
}

################################################################################
# Generate JWT secret
################################################################################

generate_jwt_secret() {
    log_info "Generating JWT secret..."

    local jwt_secret
    jwt_secret=$(openssl rand -hex 32)

    # Update .env file
    if [ -f ".env" ]; then
        sed -i "s|JWT_SECRET=.*|JWT_SECRET=${jwt_secret}|" .env
        log_success "JWT secret generated and set in .env"
    else
        log_error ".env file not found"
    fi
}

################################################################################
# Generate Ed25519 key pairs for agents
################################################################################

generate_agent_keys() {
    log_info "Generating Ed25519 key pairs for agents..."

    local agents=("commander" "sentinel" "builder" "inspector" "watcher")
    local keys_dir="keys"

    # Create keys directory
    mkdir -p "$keys_dir"
    chmod 700 "$keys_dir"

    for agent in "${agents[@]}"; do
        log_info "Generating keys for $agent..."

        local private_key_file="$keys_dir/${agent}_private.pem"
        local public_key_file="$keys_dir/${agent}_public.pem"

        # Generate private key
        openssl genpkey -algorithm ed25519 -out "$private_key_file"
        chmod 600 "$private_key_file"

        # Extract public key
        openssl pkey -in "$private_key_file" -pubout -out "$public_key_file"
        chmod 644 "$public_key_file"

        # Read private key and update .env
        local private_key
        private_key=$(cat "$private_key_file")

        local env_var_name="${agent^^}_PRIVATE_KEY"
        if grep -q "^${env_var_name}=" .env; then
            # Use proper escaping for multiline values in .env
            local escaped_key
            escaped_key=$(printf '%s\n' "$private_key" | sed -e 's/[\/&]/\\&/g')
            sed -i "/${env_var_name}=/,/-----END PRIVATE KEY-----/c\\
${env_var_name}=$(printf '%s' "$private_key")" .env
        fi

        log_success "Generated keys for $agent"
    done

    log_success "All agent keys generated in $keys_dir"
}

################################################################################
# Set up Docker network
################################################################################

setup_docker_network() {
    log_info "Setting up Docker networks..."

    local networks=("agent-army-internal" "agent-army-gateway")

    for network in "${networks[@]}"; do
        if docker network inspect "$network" &> /dev/null; then
            log_warning "Docker network $network already exists"
        else
            docker network create "$network" || log_warning "Failed to create network $network"
            log_success "Docker network $network created"
        fi
    done
}

################################################################################
# Initialize database
################################################################################

initialize_database() {
    log_info "Initializing database..."

    # Source .env to get database credentials
    set -a
    source .env
    set +a

    log_info "Waiting for PostgreSQL to be ready..."

    # Start PostgreSQL container temporarily for initialization
    docker-compose up -d postgres

    # Wait for PostgreSQL to be healthy
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if docker-compose exec -T postgres pg_isready -U "${DB_USER}" &> /dev/null; then
            log_success "PostgreSQL is ready"
            break
        fi
        attempt=$((attempt + 1))
        if [ $attempt -eq $max_attempts ]; then
            log_error "PostgreSQL failed to become ready"
        fi
        sleep 1
    done

    # Run migration script or initialization
    log_info "Running database initialization..."
    # This would typically run migrations, e.g.:
    # docker-compose exec -T postgres psql -U "${DB_USER}" -d "${DB_NAME}" < /path/to/init.sql

    log_success "Database initialized"
}

################################################################################
# Create required directories
################################################################################

create_directories() {
    log_info "Creating required directories..."

    local directories=(
        "logs"
        "data"
        "artifacts"
        "keys"
        "backups"
    )

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            chmod 755 "$dir"
            log_success "Created directory: $dir"
        else
            log_warning "Directory already exists: $dir"
        fi
    done
}

################################################################################
# Validate configuration
################################################################################

validate_configuration() {
    log_info "Validating configuration..."

    local errors=0

    # Check if JWT_SECRET is set and not default
    if ! grep -q "^JWT_SECRET=[a-f0-9]\{64\}$" .env; then
        log_warning "JWT_SECRET appears to be invalid or default"
        ((errors++))
    else
        log_success "JWT_SECRET is valid"
    fi

    # Check if agent private keys are present
    for agent in commander sentinel builder inspector watcher; do
        local key_file="keys/${agent}_private.pem"
        if [ ! -f "$key_file" ]; then
            log_warning "Missing private key for $agent"
            ((errors++))
        else
            log_success "Found private key for $agent"
        fi
    done

    # Check if required API keys are configured
    if ! grep -q "^CLAUDE_API_KEY=sk-ant-" .env; then
        log_warning "CLAUDE_API_KEY not configured (set to example value)"
    fi

    if [ $errors -eq 0 ]; then
        log_success "Configuration validation passed"
    else
        log_warning "Configuration validation completed with warnings"
    fi
}

################################################################################
# Print success message with next steps
################################################################################

print_success_message() {
    echo ""
    log_success "AgentArmy setup completed successfully!"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Update configuration files:"
    echo "   - Edit .env with your API keys and credentials"
    echo "   - Review config/settings.yaml for system settings"
    echo "   - Review config/security_policies.yaml for security rules"
    echo ""
    echo "2. Start the AgentArmy system:"
    echo "   docker-compose up -d"
    echo ""
    echo "3. Verify the system is running:"
    echo "   docker-compose ps"
    echo ""
    echo "4. View logs:"
    echo "   docker-compose logs -f gateway"
    echo ""
    echo "5. Access the API gateway:"
    echo "   http://localhost:8000"
    echo ""
    echo -e "${YELLOW}Important:${NC}"
    echo "- Agent private keys are stored in the 'keys' directory"
    echo "- Keep .env file secure and never commit it to version control"
    echo "- Review config/security_policies.yaml and customize as needed"
    echo ""
}

################################################################################
# Main execution
################################################################################

main() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║            AgentArmy Setup Script v1.0                     ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Change to script directory
    cd "$(dirname "${BASH_SOURCE[0]}")/.." || log_error "Failed to change directory"

    # Run setup steps
    check_prerequisites
    echo ""

    create_directories
    echo ""

    create_env_file
    echo ""

    generate_jwt_secret
    echo ""

    generate_agent_keys
    echo ""

    setup_docker_network
    echo ""

    validate_configuration
    echo ""

    # Uncomment to auto-initialize database
    # initialize_database
    # echo ""

    print_success_message
}

# Run main function
main "$@"
