// API functions for communicating with the Kyaro Anti-DDoS backend

import { KyaroConfig } from './config-mapper';

const API_BASE_URL = 'http://localhost:6868/api';

// API response type
export interface ApiResponse<T> {
  success: boolean;
  message: string;
  data?: T;
}

// Config update request type
export interface ConfigUpdateRequest {
  config: KyaroConfig;
}

// Fetch config from the backend
export async function fetchConfig(): Promise<KyaroConfig> {
  try {
    const response = await fetch(`${API_BASE_URL}/config`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const result: ApiResponse<KyaroConfig> = await response.json();
    
    if (!result.success || !result.data) {
      throw new Error(result.message || 'Failed to fetch configuration');
    }
    
    return result.data;
  } catch (error) {
    console.error('Error fetching config:', error);
    throw error;
  }
}

// Save config to the backend
export async function saveConfig(config: KyaroConfig): Promise<KyaroConfig> {
  try {
    const request: ConfigUpdateRequest = { config };
    
    const response = await fetch(`${API_BASE_URL}/config`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const result: ApiResponse<KyaroConfig> = await response.json();
    
    if (!result.success || !result.data) {
      throw new Error(result.message || 'Failed to save configuration');
    }
    
    return result.data;
  } catch (error) {
    console.error('Error saving config:', error);
    throw error;
  }
} 