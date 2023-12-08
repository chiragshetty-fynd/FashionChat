import requests


class PromptEnhancer:
    URL = "http://openai_hub:80/enhance"
    HEADERS = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    @staticmethod
    def enhance(prompt):
        data = {"prompt": prompt}
        response = requests.post(
            PromptEnhancer.URL, headers=PromptEnhancer.HEADERS, data=data
        )
        if response.status_code == 200:
            result = response.json()
            enhanced_prompt = result.get("enhanced_prompt", prompt)
            return enhanced_prompt
        print(f"Error: {response.status_code} - {response.text}")
        return prompt
