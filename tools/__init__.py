def tool_weather(user_prompt, memory):
    combined_query = (user_prompt + " " + memory).lower()
    if "tokyo" in combined_query:
        return "Tokyo: Sunny, 22C."
    elif "london" in combined_query:
        return "London: Raining, 14C."
    return "Location unknown."


def tool_bash_simulator(user_prompt, memory):
    user_lower = user_prompt.lower()
    if any(kw in user_lower for kw in ["list", "ls", "files", "directory", "dir"]):
        return "file1.txt\nhidden_folder\nsystem_log.csv"
    if any(kw in user_lower for kw in ["python", "version"]):
        return "Python 3.12.3"
    if any(kw in user_lower for kw in ["date", "time", "today", "now"]):
        return "2026-04-05 14:32:00 UTC"
    if any(kw in user_lower for kw in ["whoami", "user", "name"]):
        return "luka"
    if any(kw in user_lower for kw in ["pwd", "where", "path", "cwd"]):
        return "/home/luka/Programming/python/amtavla"
    if any(kw in user_lower for kw in ["disk", "space", "df", "storage"]):
        return "Filesystem      Size  Used Avail Use%\n/dev/sda1       500G  120G  380G  24%"
    return f"Simulated output for: {user_prompt}"
