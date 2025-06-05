---
title: "Nous Update #2"
date: 2025-06-05 00:00:00 -0500
categories: [Nous]
tags: [Updates, Fullstack, Development, Project]
---

This week, I have integrated firestore compatibility and user control. Currently, I have defined the user schema with the following elements: id, first_name, last_name, active_at, created, email_address, avatar, watchlist, searched. I do feel that I may need to add more to this schema but I will do this at a later data. I have built a layout widget for the app this week as well with all necessary components for navigation. For the user creation and updates, I am currently using webhooks from clerk to run a firebase function that updates the firestore database. I have added a visual below to demonstrate how it currently works.

![Architecture Update Graphic](/assets/update_2.png "Architecture Update Graphic")
