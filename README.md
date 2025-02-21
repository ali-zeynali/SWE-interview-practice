# **Practice for SWE Interview**
Welcome to the SWE-Interview-Practice! This platform is dedicated to providing concise and insightful solutions to algorithmic and data structure challenges, with a focus on Leetcode-style problems. Here, you'll find clear explanations, optimized solutions, and key insights to help you master problem-solving techniques. Whether you're a beginner honing your skills or an experienced coder preparing for technical interviews, these bite-sized solutions aim to enhance your understanding and boost your confidence. Powered by AI-assisted insights, this resource is designed to make learning algorithms both engaging and effective.

***
***
&nbsp;
## **Easy Questions**
&nbsp;

***
#### Given a string s consisting of words and spaces, return the length of the last word in the string. A word is a maximal substring consisting of non-space characters only.

![Version](https://img.shields.io/badge/String-blue)  
```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        s = s.strip()
        if len(s) == 0:
            return 0
        return len(s.split(" ")[-1])
```

***
***
&nbsp;
## **Medium Questions**
&nbsp;

***

***
#### You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position. Return true if you can reach the last index, or false otherwise.

![Version](https://img.shields.io/badge/Array-white)  
```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [0 for __ in range(n)]
        dp[0] = nums[0]
        for i in range(1, n):
            if dp[i-1] < i:
                return False
            dp[i] = max(dp[i-1], i +nums[i])
        return True
```

***
***
&nbsp;
## **Hard Questions**
&nbsp;

***

***
#### Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

![Version](https://img.shields.io/badge/Array-white)  
```python
import numpy as np
class Solution:
    def trap(self, height: List[int]) -> int:
        
        # return self.helper(height, 0, 0)
        return self.helper2(height)

    def helper(self, height, max_heightL, max_heightR):
        # Complexity is O(n log(n))
        if len(height) == 0:
            return 0
        max_val = max(height)
        if max_val <= min(max_heightL, max_heightR):
            return len(height) * min(max_heightL, max_heightR) - sum(height)
        k = np.argmax(height)
        return self.helper(height[:k], max_heightL, max_val) + self.helper(height[k+1:], max_val, max_heightR) 

    def helper2(self, height):
        # Complexity is O (n)
        n = len(height)
        left_max = float('-inf')
        right_max = float('-inf')
        left = 0
        right = n- 1
        output = 0

        while left <= right:
            if height[left] <= height[right]:
                left_max = max(left_max, height[left])
                output += left_max - height[left]
                left +=1
            else:
                right_max = max(right_max, height[right])
                output += right_max - height[right]
                right -= 1
        return output
```

