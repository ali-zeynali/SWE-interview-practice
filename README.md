# **Practice for SWE Interview**
Welcome to the SWE-Interview-Practice! This platform is dedicated to providing concise and insightful solutions to algorithmic and data structure challenges, with a focus on Leetcode-style problems. Here, you'll find clear explanations, optimized solutions, and key insights to help you master problem-solving techniques. Whether you're a beginner honing your skills or an experienced coder preparing for technical interviews, these bite-sized solutions aim to enhance your understanding and boost your confidence. Powered by AI-assisted insights, this resource is designed to make learning algorithms both engaging and effective.

***
***
&nbsp;
## **Easy Questions**
&nbsp;

***
#### Useful String functions

![Version](https://img.shields.io/badge/String-blue)  
# Useful Python String Functions

| Function        | Description                            |
|--------------- |--------------------------------|
| strip()      | Removes leading/trailing spaces |
| lstrip()     | Removes leading spaces          |
| rstrip()     | Removes trailing spaces         |
| lower()      | Converts string to lowercase    |
| upper()      | Converts string to uppercase    |
| isdigit()    | Checks if string contains only digits |
| isalpha()    | Checks if string contains only letters |
| chr()        | Converts an ASCII code to its character |
| ord()        | Converts a character to its ASCII code |


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
#### given a list (lst), sort its objecs based on: if o1['v1'] < o2['v1'] -> o1 comes before o2. If o1['v1'] == o2['v1'] and o1['v2'] < o2['v2'] -> o1 comes before o2.

![Version](https://img.shields.io/badge/Sort-gray)  
```python
class Solution:
    def customizedSort(lst):
        def myKey(o1, o2):
            if o1['v1'] < o2['v1']:
                return -1
            if o1['v1'] > o2['v1']:
                return 1
            if o1['v2'] < o2['v2']:
                return -1
            return 1
        return sorted(lst, key=cmp_to_key(myKey))
```

***
***
&nbsp;
## **Medium Questions**
&nbsp;

***

***
#### Given an integer array nums where every element appears three times except for one, which appears exactly once. Find the single element and return it. You must implement a solution with a linear runtime complexity and use only constant extra space.

![Version](https://img.shields.io/badge/Bitwise-yellow)  
```python
class Solution:
    def singleNumber(nums):
        ones, twos = 0, 0
        for num in nums:
            ones = (ones ^ num) & ~twos  # XOR num into ones and clear bits present in twos
            twos = (twos ^ num) & ~ones  # XOR num into twos and clear bits present in ones
        return ones
```


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
#### Given an array of integers citations where citations[i] is the number of citations a researcher received for their ith paper, return the researcher's h-index.

![Version](https://img.shields.io/badge/Array-white)  
```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        sortedCit = sorted(citations, reverse=True)
        hIndex = 0
        for i in range(len(sortedCit)):
            h = min(sortedCit[i], i+1)
            hIndex = max(h, hIndex)
        return hIndex
```

***
#### There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

#### You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.

#### Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique.

![Version](https://img.shields.io/badge/Array-white)  
```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        start = 0
        tank = 0
        total_tank = 0
        for i in range(n):
            tank = tank + gas[i] - cost[i]
            total_tank += gas[i] - cost[i]
            if tank < 0:
                start = i + 1
                tank = 0
        return start if total_tank >= 0 else -1
```

***
#### Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.
![Version](https://img.shields.io/badge/SlidingWindow-purple)  
```python
import numpy as np
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        totalSum = 0
        maxSum = float('-inf')
        minSum = float('inf')
        curMax = 0
        curMin = 0

        for v in nums:
            totalSum += v
            curMax = max(curMax+v, v)
            maxSum = max(maxSum, curMax)

            curMin = min(curMin+v, v)
            minSum = min(minSum, curMin)

        return maxSum if maxSum <= 0 else max(maxSum, totalSum - minSum)
```

***
#### A peak element is an element that is strictly greater than its neighbors. Given a 0-indexed integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks. Answer in O(log(n))

![Version](https://img.shields.io/badge/BinarySearch-green)  
```python
import numpy as np
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) <= 3:
            return np.argmax(nums)
        
        left = 0
        right = len(nums) - 1

        if nums[left+1] < nums[left]:
            return left
        if nums[right - 1] < nums[right]:
            return right
        
        mid = (left + right) // 2

        if (mid == 0 or nums[mid-1] < nums[mid]) and (mid == right or nums[mid] > nums[mid+1]):
            return mid
        
        if nums[mid] > max(nums[left], nums[right]):
            if nums[mid-1] > nums[mid]:
                return self.findPeakElement(nums[:mid+1])
            else:
                return self.findPeakElement(nums[mid:]) + mid
                
        else:
            if nums[left] > nums[right]:
                return self.findPeakElement(nums[:mid+1])
            else:
                return self.findPeakElement(nums[mid:]) + mid
```

***
#### You are given two integer arrays nums1 and nums2 sorted in non-decreasing order and an integer k. Define a pair (u, v) which consists of one element from the first array and one element from the second array. Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.

![Version](https://img.shields.io/badge/Heap-cyan)  
```python
import heapq
import numpy as np
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        def getRan():
            return np.random.random()*0.0001
        visited = {}
        heap = []
        heapq.heappush(heap, (nums1[0]+nums2[0] + getRan(), [0, 0]))
        visited[0] = set([])
        visited[0].add(0) 
        ans = []
        n = len(nums1)
        m = len(nums2)
        # complexity: K * log(HEAP_SIZE) ~ K * log(K) 
        while len(ans) < k:
            (v, [i, j]) = heapq.heappop(heap)
            ans.append([nums1[i] , nums2[j]])
            # opt 1: moving i
            if i < n - 1 and (i+1 not in visited or j not in visited[i+1]):
                if i+1 not in visited:
                    visited[i+1] = set([j])
                else:
                    visited[i+1].add(j)
                val = nums1[i+1] + nums2[j] + getRan()
                heapq.heappush(heap, (val, [i+1, j]))
            # opt 2: moving j 
            if j < m - 1 and (i not in visited or j+1 not in visited[i]):
                if i not in visited:
                    visited[i] = set([j+1])
                else:
                    visited[i].add(j+1)

                val = nums1[i] + nums2[j+1] + getRan()
                heapq.heappush(heap, (val, [i, j+1]))
        return ans
```

***
#### Given the array of number of possible jumps from index i, return minimum number of jumps required to reach the end of the list.

![Version](https://img.shields.io/badge/Array-white)  
```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        near = far = jumps = 0
        while far < len(nums) - 1:
            farthest = 0
            for i in range(near, far + 1):
                farthest = max(farthest, i + nums[i])
            near = far + 1
            far = farthest
            jumps += 1
        return jumps
```

***


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

*** 
#### There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.

#### You are giving candies to these children subjected to the following requirements:

* #### Each child must have at least one candy.
* #### Children with a higher rating get more candies than their neighbors.

#### Return the minimum number of candies you need to have to distribute the candies to the children.

![Version](https://img.shields.io/badge/Array-white)  
```python
class Solution:
    def candy(self, ratings: List[int]) -> int:

        n = len(ratings)
        candies = [1] * n 

        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1

        for i in range(n-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                candies[i] = max(candies[i], candies[i+1] + 1)
        return sum(candies)
```

***
#### Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

![Version](https://img.shields.io/badge/SlidingWindow-purple)  
```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        def get_left_areas(heights):
            stack = []
            areas = []
            for i, h in enumerate(heights):
                if not stack or h > stack[-1][1]:
                    stack.append([i, h])
                    areas.append(h)
                else:
                    while stack and stack[-1][1] >= h:
                        stack.pop()
                    if not stack:
                        areas.append(h*(i+1))
                    else:
                        areas.append(h*(i-stack[-1][0]))
                    stack.append([i, h])
            return areas
        if len(heights) == 0:
            return 0
        left = get_left_areas(heights)
        right = get_left_areas(heights[::-1])[::-1]
        max_area = 0
        for i in range(len(heights)):
            area = left[i] + right[i] - heights[i]
            if area > max_area:
                max_area = area
        return max_area
```