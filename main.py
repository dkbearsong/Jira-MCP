from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv, set_key
from pathlib import Path
import logging, os
from jira import JIRA
from jql_builder import build_JQL



load_dotenv()

mcp = FastMCP("jira_sentiment")
USER_AGENT = "jira_sentiment-app/1.0"

log_dir = os.path.expanduser("~/.mcp_logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "jira_mcp.log"),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.debug("MCP Server starting up...")

def get_env_var(from_mcp: bool = False):
    """ Grabs the API token and the Jira site, reqeusts it if it does not exist. """
    # Checks to see if an API token exists in the .env. If not, first checks for a .env and creates it, then requests the email address and API token to store. Skips if from MCP
    env_path = Path('.') / '.env'
    env_path.touch(exist_ok=True)

    # Prefer explicit JIRA-specific variable names; fall back for backwards compatibility
    jira_email = os.getenv("JIRA_EMAIL") or os.getenv("JIRA_USER_EMAIL") or os.getenv("USER")
    api_token = os.getenv("API_TOKEN")
    jira_site = os.getenv("JIRA_SITE")

    if (not jira_email or not api_token) and from_mcp:
        return ("API credentials not set. Please run a setup step to set JIRA_EMAIL and API_TOKEN in .env.")

    if not api_token and not from_mcp:
        user_input = input(f"Please provide the user's Jira email address: ")
        token_input = input(f"""
            Please provide the API Token for user {user_input}. Visit https://id.atlassian.com/manage-profile/security/api-tokens to create a token. 
            Ensure the scoped token has Jira access and the following classic scopes:
            read:jira-work 
            write:jira-work

        """)
        set_key(env_path, "JIRA_EMAIL", user_input)
        set_key(env_path, "API_TOKEN", token_input)
        jira_email, api_token = user_input, token_input

    if not jira_site and from_mcp:
        return ("JIRA site missing. Please set JIRA_SITE in .env (e.g., https://yourorg.atlassian.net).")

    if not jira_site and not from_mcp:
        site_input = input("Please provide the Jira site (e.g., https://yourorg.atlassian.net): ")
        set_key(env_path, "JIRA_SITE", site_input)
        jira_site = site_input

    return {
        'user': jira_email,
        'API_token': api_token,
        'Jira_site': jira_site
    }

def issue_search(instance, key, rf):
    """
    Runs the API search to get the a single tickets details

    Args:
        instance: obj containing Jira instance
        key: str containing Jira Issue key
        crit: dict containing list of fields to search for

    Return:
    """
    if rf:
        fields = ", ".join(rf)
    else:
        fields = "*all"
    issue = instance.issue(key, fields=fields)
    field_guide = get_custom_fields(instance)
    issue_dict = issue.raw

    # Remap custom field keys inside the 'fields' object to human-readable names
    fields_dict = issue_dict.get("fields", {})
    remapped_fields = {}
    for field_key, field_value in fields_dict.items():
        if isinstance(field_key, str) and field_key.startswith("customfield_"):
            field_id = field_key.split("_")[1]
            human_readable = field_guide.get(field_id, field_key)
            remapped_fields[human_readable] = field_value
        else:
            remapped_fields[field_key] = field_value

    issue_dict["fields"] = remapped_fields
    return issue_dict

def jql_search(instance, query, max_results: int = 50):
    """
    Runs the API search to get a JQL search

    Args:
        instance: obj containing Jira instance
        query: str containing JQL query to run search with

    Return:
    """
    start_at = 0
    all_issues = []
    field_guide = get_custom_fields(instance)

    while True:
        results = instance.search_issues(query, startAt=start_at, maxResults=max_results)
        if not results:
            break
        for issue in results:
            issue_dict = issue.raw
            fields_dict = issue_dict.get("fields", {})
            remapped_fields = {}
            for field_key, field_value in fields_dict.items():
                if isinstance(field_key, str) and field_key.startswith("customfield_"):
                    field_id = field_key.split("_")[1]
                    human_readable = field_guide.get(field_id, field_key)
                    remapped_fields[human_readable] = field_value
                else:
                    remapped_fields[field_key] = field_value
            issue_dict["fields"] = remapped_fields
            all_issues.append(issue_dict)
        if len(results) < max_results:
            break
        start_at += max_results

    return all_issues

def get_custom_fields(instance):
    custom_fields = {}
    all_fields = instance.fields()
    for field in all_fields:
        # Some servers may not include 'custom' key for system fields
        if field.get('custom'):
            # field['id'] is like 'customfield_10015' – we need the numeric id part to match our split
            field_id_full = field.get('id', '')
            field_id = field_id_full.split('_')[-1] if '_' in field_id_full else field_id_full
            custom_fields[field_id] = field.get('name', field_id_full)
    return custom_fields


@mcp.tool()
def jira_search(users: list, company: list, task_type: str, issue_key: str, criteria: dict, return_fields: list):
    """
    Searches a user's Jira site for issues based on given parameters and returns
    issue details for an agent to analyze.

    Args:
        users: list of users to search for
        company: list of companies to search for, if relevant
        task_type: "issue_search" for a single issue key, or "JQL" for a
            criteria-based search
        issue_key: Jira issue key when task_type == "issue_search"
        criteria: dictionary describing field-based filters for a JQL search
        return_fields: list of fields to include in the response

    JSON structure for criteria (strict):
    - Only use Jira-recognized field names that exist in the site as keys. Do
      not invent or include extra/non-Jira fields inside criteria.
    - Each field key may have either:
      1) a simple value, or
      2) an object with an operator/value (and optional function).
    - Logical grouping is supported with top-level keys "and" or "or", each
      mapping to a list of criteria dictionaries. Each dict in these lists must
      still use Jira field names as keys.

    Examples:
    {
        'reporter': {
            'operator': 'equals',
            'value': 'Dereck Bearsong'
        },
        'priority': 'Low'
    }

    {
        'and': [
            { 'reporter': { 'operator': 'equals', 'value': 'Dereck Bearsong' } },
            { 'priority': { 'operator': 'equals', 'value': 'Low' } }
        ]
    }

    Supported operators include: equals, not_equals, in, not_in, contains,
    not_contains, greater_than, less_than, is, is_not, was, was_in, was_not,
    was_not_in, changed. The optional 'function' can be used for JQL functions
    like 'currentUser()' or 'startOfDay()'.

    Return:
        dict or str: providing details about the ticket or tickets that have been requested if successful, string with error details if error occurs
    """
    env_vars = get_env_var(True)
    if isinstance(env_vars, str):
        return f"Fatal Error: {env_vars}"
    
    jira = JIRA(server=env_vars['Jira_site'], basic_auth=(env_vars['user'], env_vars['API_token']))

    if task_type == "issue_search":
        issue = issue_search(jira, issue_key, return_fields)
        return issue
    elif task_type == 'JQL':
        JQL = build_JQL(users, company, criteria)
        issues = jql_search(jira, JQL)
        return issues

@mcp.prompt()
def jira_searcher():
    """Global Instructions for running a Jira Search"""
    return f"""# Jira Searcher

    You are a Jira search agent, an agent specializing in pulling details about issues in Jira Cloud.
    Search the user's Jira site for the requested issue or issues and return details on the fields 
    specified. The user will provide either a specific issue key or a list of criteria to use in order 
    to search for these issues, and any fields they want specifically returned back.

    Return the details on the requested issues, being sure to provide specifically the requested fields
    if provided. If no specific fields to return back are provided, provide all fields.

    """


if __name__ == "__main__":
    mcp.run(transport="stdio")

# Current To-Do: update the single issue search to pull custom field names
# Future To-Do: create a write ticket and edit ticket tool.
