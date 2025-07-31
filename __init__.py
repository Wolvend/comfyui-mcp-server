# ComfyUI MCP Server
# This is a standalone MCP server, not a ComfyUI node
# The actual server runs separately via server.py

"""
ComfyUI MCP Server - Model Context Protocol interface for ComfyUI
This module provides MCP integration but doesn't register any ComfyUI nodes.
"""

# Empty node mappings since this is a service, not a node pack
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']