#!/usr/bin/env python3
"""
Code Analyzer for IRONFORGE Semantic Indexer
============================================

Deep AST-based analysis of Python source code to extract:
- Classes, functions, and methods
- Import dependencies
- Docstrings and type hints
- Complexity metrics
- Design patterns
"""

import ast
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import re
import logging


class CodeAnalyzer:
    """
    AST-based Python code analyzer for semantic extraction.
    
    Provides comprehensive analysis including:
    - Structure extraction (classes, functions, imports)
    - Complexity metrics
    - Type hint analysis
    - Pattern detection
    - Documentation extraction
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ironforge.indexer.analyzer')
    
    def analyze_file(self, file_path: Path, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Analyze a single Python file using AST parsing.
        
        Args:
            file_path: Path to Python file to analyze
            quick_mode: If True, perform lightweight analysis
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST
            tree = ast.parse(source_code, filename=str(file_path))
            
            # Extract components
            analysis = {
                'file_path': str(file_path),
                'imports': self._extract_imports(tree),
                'classes': self._extract_classes(tree, source_code, quick_mode),
                'functions': self._extract_functions(tree, source_code, quick_mode),
                'constants': self._extract_constants(tree),
                'decorators': self._extract_decorators(tree),
                'metrics': self._calculate_metrics(tree, source_code),
                'patterns': self._detect_patterns(tree, source_code) if not quick_mode else [],
                'docstring': self._extract_module_docstring(tree)
            }
            
            return analysis
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
            return {'error': f'Syntax error: {e}', 'file_path': str(file_path)}
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return {'error': str(e), 'file_path': str(file_path)}
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract import statements and dependencies."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'line_number': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'level': node.level,
                        'line_number': node.lineno
                    })
        
        return imports
    
    def _extract_classes(self, tree: ast.AST, source_code: str, quick_mode: bool) -> List[Dict[str, Any]]:
        """Extract class definitions and their metadata."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'docstring': self._safe_get_docstring(node),
                    'base_classes': [self._get_base_class_name(base) for base in node.bases],
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'is_abstract': self._is_abstract_class(node),
                    'is_exception': self._is_exception_class(node)
                }
                
                if not quick_mode:
                    class_info.update({
                        'methods': self._extract_methods(node, source_code),
                        'class_variables': self._extract_class_variables(node),
                        'properties': self._extract_properties(node),
                        'complexity': self._calculate_class_complexity(node)
                    })
                
                classes.append(class_info)
        
        return classes
    
    def _extract_functions(self, tree: ast.AST, source_code: str, quick_mode: bool) -> List[Dict[str, Any]]:
        """Extract function definitions and their metadata."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                func_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'docstring': self._safe_get_docstring(node),
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'is_generator': self._is_generator(node),
                    'is_private': node.name.startswith('_'),
                    'parameters': self._extract_parameters(node)
                }
                
                if not quick_mode:
                    func_info.update({
                        'complexity': self._calculate_function_complexity(node),
                        'return_annotation': self._get_annotation(node.returns),
                        'calls': self._extract_function_calls(node),
                        'local_variables': self._extract_local_variables(node)
                    })
                
                functions.append(func_info)
        
        return functions
    
    def _extract_methods(self, class_node: ast.ClassDef, source_code: str) -> List[Dict[str, Any]]:
        """Extract methods from a class definition."""
        methods = []
        
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'docstring': self._safe_get_docstring(node),
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'is_property': self._is_property(node),
                    'is_classmethod': self._is_classmethod(node),
                    'is_staticmethod': self._is_staticmethod(node),
                    'is_private': node.name.startswith('_'),
                    'is_dunder': node.name.startswith('__') and node.name.endswith('__'),
                    'parameters': self._extract_parameters(node),
                    'complexity': self._calculate_function_complexity(node)
                }
                
                methods.append(method_info)
        
        return methods
    
    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract module-level constants."""
        constants = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            'name': target.id,
                            'line_number': node.lineno,
                            'type': self._infer_constant_type(node.value)
                        })
        
        return constants
    
    def _extract_decorators(self, tree: ast.AST) -> Set[str]:
        """Extract all unique decorators used in the file."""
        decorators = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    decorators.add(self._get_decorator_name(decorator))
        
        return list(decorators)
    
    def _calculate_metrics(self, tree: ast.AST, source_code: str) -> Dict[str, Any]:
        """Calculate various code metrics."""
        lines = source_code.split('\n')
        
        return {
            'lines_of_code': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'docstring_lines': self._count_docstring_lines(tree),
            'complexity_total': self._calculate_total_complexity(tree),
            'class_count': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            'function_count': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            'import_count': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
        }
    
    def _detect_patterns(self, tree: ast.AST, source_code: str) -> List[Dict[str, Any]]:
        """Detect common design patterns in the code."""
        patterns = []
        
        # Factory Pattern Detection
        if self._detect_factory_pattern(tree):
            patterns.append({
                'pattern': 'Factory',
                'confidence': 'high',
                'description': 'Creates objects without specifying exact class'
            })
        
        # Builder Pattern Detection
        if self._detect_builder_pattern(tree):
            patterns.append({
                'pattern': 'Builder',
                'confidence': 'high',
                'description': 'Constructs complex objects step by step'
            })
        
        # Singleton Pattern Detection
        if self._detect_singleton_pattern(tree):
            patterns.append({
                'pattern': 'Singleton',
                'confidence': 'medium',
                'description': 'Ensures only one instance exists'
            })
        
        # Strategy Pattern Detection
        if self._detect_strategy_pattern(tree):
            patterns.append({
                'pattern': 'Strategy',
                'confidence': 'medium',
                'description': 'Encapsulates algorithms and makes them interchangeable'
            })
        
        # Container/Dependency Injection Pattern
        if self._detect_container_pattern(tree):
            patterns.append({
                'pattern': 'Container/DI',
                'confidence': 'high',
                'description': 'Dependency injection container pattern'
            })
        
        return patterns
    
    # Helper methods for pattern detection
    def _detect_factory_pattern(self, tree: ast.AST) -> bool:
        """Detect factory pattern by looking for factory method signatures."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if 'create' in node.name.lower() or 'factory' in node.name.lower():
                    return True
            elif isinstance(node, ast.ClassDef):
                if 'factory' in node.name.lower():
                    return True
        return False
    
    def _detect_builder_pattern(self, tree: ast.AST) -> bool:
        """Detect builder pattern by looking for method chaining."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if 'builder' in node.name.lower():
                    return True
                # Check for method chaining pattern
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        for stmt in method.body:
                            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
                                if stmt.value.id == 'self':
                                    return True
        return False
    
    def _detect_singleton_pattern(self, tree: ast.AST) -> bool:
        """Detect singleton pattern."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for __new__ method or _instance variable
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__new__':
                        return True
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and 'instance' in target.id.lower():
                                return True
        return False
    
    def _detect_strategy_pattern(self, tree: ast.AST) -> bool:
        """Detect strategy pattern by looking for strategy interfaces."""
        has_strategy_interface = False
        has_concrete_strategies = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if 'strategy' in node.name.lower() or 'algorithm' in node.name.lower():
                    has_strategy_interface = True
                # Check for multiple classes inheriting from strategy
                for base in node.bases:
                    if isinstance(base, ast.Name) and 'strategy' in base.id.lower():
                        has_concrete_strategies = True
        
        return has_strategy_interface and has_concrete_strategies
    
    def _detect_container_pattern(self, tree: ast.AST) -> bool:
        """Detect dependency injection container pattern."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if 'container' in node.name.lower() or 'injector' in node.name.lower():
                    return True
                # Look for get/register methods typical of DI containers
                method_names = [m.name.lower() for m in node.body if isinstance(m, ast.FunctionDef)]
                if 'get' in method_names and 'register' in method_names:
                    return True
        return False
    
    # Utility methods
    def _get_base_class_name(self, base: ast.expr) -> str:
        """Extract base class name from AST node."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self._get_base_class_name(base.value)}.{base.attr}"
        return str(base)
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_decorator_name(decorator.value)}.{decorator.attr}"
        return str(decorator)
    
    def _get_annotation(self, annotation: Optional[ast.expr]) -> Optional[str]:
        """Extract type annotation string."""
        if annotation is None:
            return None
        
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{self._get_annotation(annotation.value)}.{annotation.attr}"
        
        return ast.unparse(annotation) if hasattr(ast, 'unparse') else str(annotation)
    
    def _extract_parameters(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters with type hints."""
        params = []
        args = func_node.args
        
        # Regular arguments
        for i, arg in enumerate(args.args):
            param_info = {
                'name': arg.arg,
                'annotation': self._get_annotation(arg.annotation),
                'default': None,
                'kind': 'positional'
            }
            
            # Check for default values
            defaults_offset = len(args.args) - len(args.defaults)
            if i >= defaults_offset:
                default_index = i - defaults_offset
                param_info['default'] = ast.unparse(args.defaults[default_index]) if hasattr(ast, 'unparse') else str(args.defaults[default_index])
            
            params.append(param_info)
        
        # *args
        if args.vararg:
            params.append({
                'name': args.vararg.arg,
                'annotation': self._get_annotation(args.vararg.annotation),
                'kind': 'var_positional'
            })
        
        # **kwargs
        if args.kwarg:
            params.append({
                'name': args.kwarg.arg,
                'annotation': self._get_annotation(args.kwarg.annotation),
                'kind': 'var_keyword'
            })
        
        return params
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_class_complexity(self, class_node: ast.ClassDef) -> int:
        """Calculate total complexity of a class."""
        total_complexity = 0
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                total_complexity += self._calculate_function_complexity(node)
        
        return total_complexity
    
    def _calculate_total_complexity(self, tree: ast.AST) -> int:
        """Calculate total cyclomatic complexity of the module."""
        total_complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_complexity += self._calculate_function_complexity(node)
        
        return total_complexity
    
    def _extract_module_docstring(self, tree: ast.AST) -> Optional[str]:
        """Extract module-level docstring."""
        try:
            return ast.get_docstring(tree)
        except (TypeError, AttributeError):
            # Fallback for cases where ast.get_docstring fails
            if isinstance(tree, ast.Module) and tree.body:
                first_stmt = tree.body[0]
                if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                    if isinstance(first_stmt.value.value, str):
                        return first_stmt.value.value
            return None
    
    def _count_docstring_lines(self, tree: ast.AST) -> int:
        """Count lines in all docstrings."""
        docstring_lines = 0
        
        for node in ast.walk(tree):
            try:
                docstring = ast.get_docstring(node)
                if docstring:
                    docstring_lines += len(docstring.split('\n'))
            except (TypeError, AttributeError):
                # Skip nodes that can't have docstrings
                continue
        
        return docstring_lines
    
    def _is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is a method (inside a class)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    def _is_generator(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is a generator."""
        for node in ast.walk(func_node):
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                return True
        return False
    
    def _is_abstract_class(self, class_node: ast.ClassDef) -> bool:
        """Check if class is abstract."""
        # Look for ABC inheritance or @abstractmethod decorators
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in ('ABC', 'AbstractBase'):
                return True
        
        for node in ast.walk(class_node):
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and 'abstract' in dec.id.lower():
                        return True
        
        return False
    
    def _is_exception_class(self, class_node: ast.ClassDef) -> bool:
        """Check if class is an exception."""
        for base in class_node.bases:
            if isinstance(base, ast.Name) and 'Exception' in base.id:
                return True
        return False
    
    def _is_property(self, method_node: ast.FunctionDef) -> bool:
        """Check if method is a property."""
        return any(
            isinstance(dec, ast.Name) and dec.id == 'property'
            for dec in method_node.decorator_list
        )
    
    def _is_classmethod(self, method_node: ast.FunctionDef) -> bool:
        """Check if method is a classmethod."""
        return any(
            isinstance(dec, ast.Name) and dec.id == 'classmethod'
            for dec in method_node.decorator_list
        )
    
    def _is_staticmethod(self, method_node: ast.FunctionDef) -> bool:
        """Check if method is a staticmethod."""
        return any(
            isinstance(dec, ast.Name) and dec.id == 'staticmethod'
            for dec in method_node.decorator_list
        )
    
    def _extract_class_variables(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class-level variables."""
        variables = []
        
        for node in class_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append({
                            'name': target.id,
                            'line_number': node.lineno,
                            'type': self._infer_constant_type(node.value)
                        })
        
        return variables
    
    def _extract_properties(self, class_node: ast.ClassDef) -> List[str]:
        """Extract property names from class."""
        properties = []
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if self._is_property(node):
                    properties.append(node.name)
        
        return properties
    
    def _extract_function_calls(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract function calls made within a function."""
        calls = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(f"{ast.unparse(node.func.value) if hasattr(ast, 'unparse') else 'obj'}.{node.func.attr}")
        
        return list(set(calls))  # Remove duplicates
    
    def _extract_local_variables(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract local variable names."""
        variables = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)
        
        return list(set(variables))  # Remove duplicates
    
    def _infer_constant_type(self, value_node: ast.expr) -> str:
        """Infer the type of a constant value."""
        if isinstance(value_node, ast.Constant):
            return type(value_node.value).__name__
        elif isinstance(value_node, ast.List):
            return 'list'
        elif isinstance(value_node, ast.Dict):
            return 'dict'
        elif isinstance(value_node, ast.Set):
            return 'set'
        elif isinstance(value_node, ast.Tuple):
            return 'tuple'
        elif isinstance(value_node, ast.Name):
            return 'reference'
        else:
            return 'unknown'
    
    def _safe_get_docstring(self, node: ast.AST) -> Optional[str]:
        """Safely extract docstring from AST node."""
        try:
            return ast.get_docstring(node)
        except (TypeError, AttributeError):
            # Fallback for cases where ast.get_docstring fails
            if hasattr(node, 'body') and node.body:
                first_stmt = node.body[0]
                if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                    if isinstance(first_stmt.value.value, str):
                        return first_stmt.value.value
            return None