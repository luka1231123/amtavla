from datetime import datetime


def tool_bash_simulator(user_prompt, memory):
    user_lower = user_prompt.lower()
    if any(kw in user_lower for kw in ["list", "ls", "files", "directory", "dir"]):
        return "file1.txt\nhidden_folder\nsystem_log.csv"
    if any(kw in user_lower for kw in ["python", "version"]):
        return "Python 3.12.3"
    if any(kw in user_lower for kw in ["date", "time", "today", "now"]):
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    if any(kw in user_lower for kw in ["whoami", "user", "name"]):
        return "luka"
    if any(kw in user_lower for kw in ["pwd", "where", "path", "cwd"]):
        return "/home/luka/Programming/python/amtavla"
    if any(kw in user_lower for kw in ["disk", "space", "df", "storage"]):
        return "Filesystem      Size  Used Avail Use%\n/dev/sda1       500G  120G  380G  24%"
    return f"Simulated output for: {user_prompt}"


def tool_weather(user_prompt, memory):
    prompt = user_prompt.lower()
    if "tokyo" in prompt:
        return "Tokyo weather: Sunny, 22C"
    if "london" in prompt:
        return "London weather: Cloudy, 14C"
    return "Weather for that location is unknown."
