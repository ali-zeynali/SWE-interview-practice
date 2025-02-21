# **Practice for SWE Interview**
Welcome to the SWE-Interview-Practice! This platform is dedicated to providing concise and insightful solutions to algorithmic and data structure challenges, with a focus on Leetcode-style problems. Here, you'll find clear explanations, optimized solutions, and key insights to help you master problem-solving techniques. Whether you're a beginner honing your skills or an experienced coder preparing for technical interviews, these bite-sized solutions aim to enhance your understanding and boost your confidence. Powered by AI-assisted insights, this resource is designed to make learning algorithms both engaging and effective.

***
***
&nbsp;
## **Easy Questions**
&nbsp;

***
#### Given a string s consisting of words and spaces, return the length of the last word in the string.
A word is a maximal substring consisting of non-space characters only.
![Version](https://img.shields.io/badge/String-blue)  
```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        s = s.strip()
        if len(s) == 0:
            return 0
        return len(s.split(" ")[-1])
```