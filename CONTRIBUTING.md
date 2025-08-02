# Contributing to ComfyUI MCP Server

We love your input! We want to make contributing to ComfyUI MCP Server as easy and transparent as possible.

## Ways to Contribute

1. **Report bugs** - Use GitHub issues
2. **Discuss ideas** - Start a discussion
3. **Submit fixes** - Open a pull request
4. **Propose features** - Create an issue first
5. **Improve docs** - Documentation PRs welcome
6. **Share workflows** - Add to examples

## Development Process

1. Fork the repo and create your branch from `main`
2. Add tests if you've added code
3. Ensure the test suite passes
4. Make sure your code follows the style guide
5. Issue your pull request!

## Code Style

- Python 3.10+ type hints
- Black formatting
- Clear docstrings
- Meaningful variable names

## Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_generation.py

# Run with coverage
python -m pytest --cov=. tests/
```

## Pull Request Process

1. Update CHANGELOG.md with your changes
2. Update TUTORIAL.md if adding features
3. Add yourself to contributors in the PR
4. The PR will be merged once approved

## Tool Development Guide

### Adding a New Tool

```python
@mcp.tool()
def your_new_tool(
    required_param: str,
    optional_param: int = 10
) -> Dict[str, Any]:
    """Brief description of what the tool does
    
    Args:
        required_param: Description
        optional_param: Description (default: 10)
        
    Returns:
        Dictionary containing result
    """
    try:
        # Implementation
        result = do_something(required_param)
        
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error in your_new_tool: {e}")
        return {"error": str(e)}
```

### Best Practices

1. **Error Handling**: Always catch and return errors gracefully
2. **Logging**: Use appropriate log levels
3. **Documentation**: Clear docstrings are mandatory
4. **Type Hints**: Use type hints for all parameters
5. **Testing**: Add tests for new tools

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.

## Questions?

Feel free to open an issue or start a discussion!