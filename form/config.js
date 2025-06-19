class ConfigLoader {
    constructor() {
        this.config = {};
        this.loaded = false;
    }

    async loadConfig() {
        try {
            // Intentar cargar desde .env
            const response = await fetch('.env');
            if (response.ok) {
                const envContent = await response.text();
                this.parseEnvContent(envContent);
                console.log('✅ Configuración cargada desde .env');
            } else {
                throw new Error('No se pudo cargar .env');
            }
        } catch (error) {
            console.warn('⚠️ No se pudo cargar .env, usando configuración por defecto');
            this.setDefaults();
        }
        
        this.loaded = true;
        return this.config;
    }

    parseEnvContent(content) {
        const lines = content.split('\n');
        lines.forEach(line => {
            line = line.trim();
            // Ignorar comentarios y líneas vacías
            if (line && !line.startsWith('#')) {
                const [key, ...valueParts] = line.split('=');
                if (key && valueParts.length > 0) {
                    const value = valueParts.join('=').trim();
                    // Remover comillas si existen
                    const cleanValue = value.replace(/^["']|["']$/g, '');
                    this.config[key.trim()] = cleanValue;
                }
            }
        });
    }

    setDefaults() {
        this.config = {
            API_URL: 'https://flashpractica.onrender.com',
            DEBUG: 'false',
        };
    }

    get(key, defaultValue = null) {
        return this.config[key] || defaultValue;
    }

    getApiUrl() {
        return this.get('API_URL', 'https://flashpractica.onrender.com');
    }

    isDebug() {
        return this.get('DEBUG', 'false').toLowerCase() === 'true';
    }
}

// Crear instancia global
window.configLoader = new ConfigLoader();