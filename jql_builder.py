def build_JQL(u: list, co: list, crit: dict):
    """
    Build a JQL query from users, companies, and criteria dictionaries.
    
    Parameters:
    -----------
    users : list, optional
        List of usernames to search for (searches in assignee, reporter, etc.)
    companies : list, optional
        List of company names to search for
    criteria : dict, optional
        Dictionary of search criteria with structure:
        {
            'field_name': {
                'operator': 'equals|not_equals|in|not_in|contains|not_contains|greater_than|less_than|...',
                'value': 'single_value' or ['list', 'of', 'values'],
                'function': 'optional_jql_function()'  # e.g., 'currentUser()', 'startOfDay()'
            },
            # OR simple format:
            'field_name': 'simple_value',
            
            # OR nested with logical operators:
            'and': [...],  # List of criteria dicts to AND together
            'or': [...]    # List of criteria dicts to OR together
        }
    
    Returns:
    --------
    str : JQL query string

    """

    OPERATOR_MAP = {
        'equals': '=',
        'not_equals': '!=',
        'greater_than': '>',
        'greater_than_equals': '>=',
        'less_than': '<',
        'less_than_equals': '<=',
        'in': 'IN',
        'not_in': 'NOT IN',
        'contains': '~',
        'not_contains': '!~',
        'is': 'IS',
        'is_not': 'IS NOT',
        'was': 'WAS',
        'was_in': 'WAS IN',
        'was_not': 'WAS NOT',
        'was_not_in': 'WAS NOT IN',
        'changed': 'CHANGED'
    }
    
    def format_value(value):
        """Format a value for JQL (add quotes if needed)."""
        if value is None:
            return 'EMPTY'
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(value)
        # Check if it's a JQL function (contains parentheses)
        if isinstance(value, str) and '(' in value and ')' in value:
            return value
        # Quote strings that contain spaces or special characters
        if isinstance(value, str):
            if ' ' in value or ',' in value or any(c in value for c in [':', '-', '/']):
                return f'"{value}"'
            return value
        return f'"{value}"'
    
    def format_value_list(values):
        """Format a list of values for JQL IN/NOT IN operators."""
        formatted = [format_value(v) for v in values]
        return f"({', '.join(formatted)})"
    
    def build_condition(field, config):
        """Build a single JQL condition from field and config."""
        # Handle simple value (string/number directly assigned)
        if not isinstance(config, dict):
            return f'{field} = {format_value(config)}'
        
        # Handle function-based value
        if 'function' in config:
            operator = OPERATOR_MAP.get(config.get('operator', 'equals'), '=')
            return f'{field} {operator} {config["function"]}'
        
        # Handle regular field with operator and value
        operator_key = config.get('operator', 'equals')
        operator = OPERATOR_MAP.get(operator_key, operator_key)
        value = config.get('value')
        
        # Handle list values for IN/NOT IN operators
        if isinstance(value, list) and operator.upper() in ['IN', 'NOT IN', 'WAS IN', 'WAS NOT IN']:
            return f'{field} {operator} {format_value_list(value)}'
        
        # Handle EMPTY/NULL checks
        if value is None or (isinstance(value, str) and value.upper() in ['EMPTY', 'NULL']):
            return f'{field} {operator} EMPTY'
        
        # Handle regular single value
        return f'{field} {operator} {format_value(value)}'
    
    def build_criteria_recursive(criteria_dict):
        """Recursively build JQL from criteria dictionary."""
        if not criteria_dict:
            return ''
        
        conditions = []
        
        # Handle logical operators (AND/OR)
        if 'and' in criteria_dict:
            sub_conditions = [build_criteria_recursive(c) for c in criteria_dict['and']]
            sub_conditions = [c for c in sub_conditions if c]  # Filter empty
            if sub_conditions:
                conditions.append(f"({' AND '.join(sub_conditions)})")
        
        if 'or' in criteria_dict:
            sub_conditions = [build_criteria_recursive(c) for c in criteria_dict['or']]
            sub_conditions = [c for c in sub_conditions if c]  # Filter empty
            if sub_conditions:
                conditions.append(f"({' OR '.join(sub_conditions)})")
        
        # Handle regular field conditions
        for field, config in criteria_dict.items():
            if field not in ['and', 'or']:
                conditions.append(build_condition(field, config))
        
        return ' AND '.join(conditions) if conditions else ''
    
    # Build the query parts
    query_parts = []
    
    # Add users clause (search in assignee by default)
    if u:
        if len(u) == 1:
            query_parts.append(f'assignee = {format_value(u[0])}')
        else:
            query_parts.append(f'assignee IN {format_value_list(u)}')
    
    # Add companies clause (assuming a 'company' custom field exists)
    if co:
        if len(co) == 1:
            query_parts.append(f'company = {format_value(co[0])}')
        else:
            query_parts.append(f'company IN {format_value_list(co)}')
    
    # Add criteria
    if crit:
        criteria_query = build_criteria_recursive(crit)
        if criteria_query:
            query_parts.append(criteria_query)
    
    # Combine all parts with AND
    jql_query = ' AND '.join(query_parts)
    
    return jql_query if jql_query else ''

# Build a main condition
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("JQL Query Builder - Interactive Mode")
    print("=" * 60)
    print()

    print("Enter users (comma-separated, or press Enter to skip):")
    print("Example: john.doe, jane.smith")
    users_input = input("> ").strip()
    users = [u.strip() for u in users_input.split(',')] if users_input else None
    print()

    print("Enter companies (comma-separated, or press Enter to skip):")
    print("Example: Acme Corp, TechCo")
    companies_input = input("> ").strip()
    companies = [c.strip() for c in companies_input.split(',')] if companies_input else None
    print()

    print("Enter criteria as JSON (or press Enter to skip):")
    print()
    print("Simple example:")
    print('  {"status": "Open", "project": "PROJ"}')
    print()
    print("With operators:")
    print('  {"status": {"operator": "in", "value": ["Open", "In Progress"]}}')
    print()
    print("With AND/OR logic:")
    print('  {"and": [{"project": "PROJ"}, {"status": "Open"}]}')
    print()
    print("Available operators: equals, not_equals, in, not_in, contains,")
    print("  not_contains, greater_than, less_than, is, is_not")
    print()
    
    criteria_input = input("> ").strip()
    criteria = None
    if criteria_input:
        try:
            criteria = json.loads(criteria_input)
        except json.JSONDecodeError as e:
            print(f"\n⚠ Invalid JSON: {e}")
            print("Continuing without criteria...\n")
    
    # Build and display the query
    print()
    print("=" * 60)
    print("Generated JQL Query:")
    print("=" * 60)
    
    jql = build_JQL(users, companies, criteria)
    
    if jql:
        print(jql)
    else:
        print("(empty query - no parameters provided)")
    
    print()
    print("=" * 60)
