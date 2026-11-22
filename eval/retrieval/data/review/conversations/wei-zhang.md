# Wei Zhang

6 conversations. Relationship: data analyst at Harborline. Person id `758ccbe9-3cc0-50f0-a302-17beddcbb6d5`.

## 2026-01-20 11:58 UTC · The late-scan label (part 1)

`b8fcab3c-2257-5428-9c3c-056749b16020`

**Summary (70 words):** Riley discussed a shipment labeled as 'Late shipments' where the delivery was on time but the system record arrived late. Wei identified the issue: the label doesn’t distinguish between scan time and when the system received the record. They reviewed a row with 9:50 a.m. scan vs 10:12 a.m. system receipt, noting the discrepancy. Riley planned to annotate the printout; Wei keeps two timestamps in a notebook to avoid confusion.

**Key facts:** On January 20, the dashboard summary is labeled “Late shipments.” | One shipment’s delivery scan shows 9:50 a.m., before its 10:00 a.m. expected delivery time. | The scan record arrived in the system at 10:12 a.m., and the shipment was counted in the summary.

**Minor details:** Wei has a small blue notebook open beside the monitor. | Riley marks the example on a printout with a pencil.

Wei: "That number includes a shipment that was actually delivered on time."  
Me: "In the 'Late shipments' summary?"  
Wei: "Yeah. That's what the dashboard calls it on January 20, anyway."  
Me: "Okay, show me which one."  
Wei: "This row here. The expected delivery time was 10:00 a.m."  
Me: "And the delivery scan says 9:50 a.m."  
Wei: "Right. But the scan record arrived in the system at 10:12 a.m."  
Me: "So the shipment made it, but the record didn't show up until after the deadline?"  
Wei: "Exactly. And that shipment was counted in the summary."  
Me: "Ah. I'm going to mark that example on my printout with this pencil."  
Wei: "The distinction is the scan time versus the time the system received the record."  
Me: "But if I just see 'Late shipments,' I'd assume the delivery itself was late."  
Wei: "I would too. In this case, the number is picking up a late-arriving scan record."  
Me: "Does it tell me that anywhere in the summary?"  
Wei: "Not in the label. It just gives you the count."  
Me: "That's a pretty important thing to leave out."  
Wei: "Yeah. The spreadsheet is technically very confident."  
Me: "Is that what you've got written in the little blue notebook beside your monitor?"  
Wei: "The two timestamps, yes. I wrote them down so I wouldn't confuse which one was which."  
Me: "Good, because 9:50 and 10:12 tell very different stories."  
Wei: "They do. One says the delivery was on time; the other says the record arrived later."  

---

## 2026-01-26 14:54 UTC · A counter out of commission (part 1)

`75b9782c-d145-5f33-97a9-d589ec975873`

**Summary (68 words):** Riley discussed the kitchen renovation planned to start March 2 at the Greenwood townhouse. They will replace the counters and the sink. For the duration, Mei and Wei will wash dishes in the bathroom, using a blue plastic dishpan. A takeout menu was placed on the fridge as a backup reference. Wei noted the situation is a temporary, imperfect workaround and preferred not to calculate expected dish volumes.

**Key facts:** The renovation is scheduled to start on March 2. | The worn counters and sink will be replaced. | Wei and Mei expect to wash dishes in the bathroom for a few weeks.

**Minor details:** Wei has picked out a blue plastic dishpan. | Mei has put a takeout menu on the fridge.

Me: "Hey, how do things stand with the kitchen renovation?"  
Wei: "It's scheduled to start on March 2, at our townhouse in Greenwood."  
Me: "That's coming up. What are you having done?"  
Wei: "We're replacing the counters and the sink. They're pretty worn at this point."  
Me: "So what do you do for a sink while that's happening?"  
Wei: "Mei and I are expecting to wash dishes in the bathroom for a few weeks."  
Me: "Oh, that's going to be a little awkward."  
Wei: "Yeah. I've picked out a blue plastic dishpan, at least. It feels like a very small solution to a fairly large problem."  
Me: "Still, probably better than trying to balance everything in the bathroom sink."  
Wei: "That's the theory. Mei's also put a takeout menu on the fridge."  
Me: "Good backup for when you can't face another round of bathroom dishes."  
Wei: "Exactly. I'm trying not to calculate how many dishes we'll wash. Some numbers are better left alone."  

---

## 2026-02-27 08:50 UTC · The late-scan label (part 2)

`d6b86292-b1d7-5051-8a6e-efbade4bc1f4`

**Summary (82 words):** Riley discussed the timestamp check, clarifying that the summary uses the receipt (system arrival) timestamp, not the delivery scan time. Example: a 9:50 delivery scan with a 10:12 record arrival is counted as late. They agreed on the label: 'Shipments with delivery scans received after expected delivery time.' The explanation should state the count does not imply the shipment itself was late, and that 'received' refers to the system receiving the scan. Action: update the February 19 document to reflect this clarification.

**Key facts:** By February 3, Wei checked the underlying scan and receipt timestamps and confirmed that the summary uses the receipt timestamp. | On February 19, Riley brought Wei the proposed label “Shipments with delivery scans received after expected delivery time.” | Riley’s accompanying explanation says the count does not establish whether the shipment itself arrived late.

**Minor details:** Riley has the wording open in a document with a pale green header. | Wei pauses to move a cold mug of tea away from the keyboard.

Wei: "Morning, Riley. It's a bit chilly in here today."  
Me: "Morning. Yeah, I was hoping it'd warm up after I got to my desk."  
Wei: "My tea had the same hope. Hang on, I'm moving this cold mug of tea away from the keyboard."  
Me: "Probably wise. I wanted to ask about the timestamp check."  
Wei: "Right. By February 3, I'd checked the underlying scan and receipt timestamps. The summary uses the receipt timestamp."  
Me: "So it's when the scan record arrives in the system, not the time on the delivery scan itself?"  
Wei: "Exactly. In the example we looked at, the delivery scan showed 9:50 a.m., before the 10:00 a.m. expected delivery time."  
Me: "But the record arrived at 10:12 a.m., so it got counted."  
Wei: "Yes. Calling the summary 'Late shipments' makes that sound like a conclusion about the shipment itself. The receipt timestamp doesn't establish that."  
Me: "That's what I was trying to address in the wording I brought you on February 19."  
Wei: "The longer label?"  
Me: "Yeah. I've got the document open here, the one with the pale green header. The proposed label is 'Shipments with delivery scans received after expected delivery time.'"  
Wei: "That names the thing we're actually counting. It's long, but it's precise."  
Me: "And the explanation says the count does not establish whether the shipment itself arrived late."  
Wei: "Good. A record can arrive late even when the scan time is before the expected delivery time. Those are different timestamps."  
Me: "I was worried people might read 'received' as the customer receiving the shipment."  
Wei: "Mm. The explanation could make it clear that the system receives the scan record. Then 'received' has somewhere to point."  
Me: "That makes sense. I'll make sure the explanation says that."  

---

## 2026-03-10 10:06 UTC · The late-scan label (part 3)

`eb5423b6-cb31-57d7-9653-fba74c7e3727`

**Summary (68 words):** Riley and Wei reviewed the staged shipment dashboard wording. They confirmed the summary label and noted the explanation about late shipments does not imply actual arrival lateness. They identified a misleading tooltip on the March 10 page that said 'Delivered late.' and agreed to replace it with 'Delivery scan received after expected delivery time' for consistency. Action: update tooltip text and annotate the margin note to resolve contradictions.

**Key facts:** On March 10, the staged tooltip still says “Delivered late.” | Riley will replace that tooltip text so it describes when the scan record was received.

**Minor details:** The staging page takes two refreshes to load. | Wei makes a note in the margin with a black pen.

Wei: "Riley, can we look at the staged shipment dashboard? I want to check the wording."  
Me: "Sure. It's loading... still loading."  
Wei: "There it is. That took 2 refreshes to load."  
Me: "A promising start. Okay, the summary has the label we discussed."  
Wei: "Yes, 'Shipments with delivery scans received after expected delivery time.' That matches the receipt timestamp the count uses."  
Me: "And the explanation still says the count doesn't establish whether the shipment itself arrived late."  
Wei: "Right. But look at the tooltip. On the March 10 staging page, it still says 'Delivered late.'"  
Me: "Oh, you're right. That makes it sound like we know the shipment was late."  
Wei: "Exactly. We know when the scan record was received, not necessarily when the shipment arrived."  
Me: "I'll replace the tooltip text so it describes when the scan record was received."  
Wei: "Good. Something like 'Delivery scan received after expected delivery time' would be more precise."  
Me: "Yeah, that keeps it consistent with the summary. I'll use that."  
Wei: "I'll make a note in the margin with my black pen: tooltip still says 'Delivered late.'"  
Me: "Thanks for catching it. The long label doesn't help much if the tooltip contradicts it."  

---

## 2026-03-18 12:56 UTC · The late-scan label (part 4)

`f507446e-e7ff-5f46-9307-a1db05cb1dd6`

**Summary (83 words):** Riley discussed dashboard wording for March 18: label updated to 'Shipments with delivery scans received after expected delivery time' and tooltip changed to 'Delivery scan record received after expected delivery time.' The calculation compares receipt timestamp to expected time, not delivery scan time. They reviewed 3 test shipments; 2 are counted due to scans after expected times, with one not counted. Clarified that a shipment can be on-time yet received after the expected time. Riley will close the tab; no further actions noted.

**Key facts:** On March 18, Riley and Wei verify the updated label and tooltip against the calculation. | Their check uses three test shipments, two of which are included in the summary count.

**Minor details:** Riley shares the screen from a corner desk. | Wei closes a browser tab after the check.

Wei: "Hi, Riley. How's your Wednesday going?"  
Me: "Hey, Wei. Pretty good. Nice to take a breath after lunch."  
Wei: "Agreed. My spreadsheet and I needed some time apart."  
Me: "I'm sharing my screen from the corner desk. Can you see the updated dashboard?"  
Wei: "Yep. So, for our March 18 check, let's compare the wording with the calculation."  
Me: "The label now says 'Shipments with delivery scans received after expected delivery time.'"  
Wei: "And the calculation compares the scan record's receipt timestamp with the expected delivery time, not the delivery scan timestamp."  
Me: "Right. The explanation also says the count doesn't establish whether the shipment itself arrived late."  
Wei: "Good. Let's look at the 3 test shipments. The summary count is 2."  
Me: "Those 2 have scan records received after their expected delivery times. The other 1 doesn't."  
Wei: "Exactly. And a scan can show an on-time delivery but still be received after the expected time. That shipment would be counted."  
Me: "Here's the tooltip I changed. It says 'Delivery scan record received after expected delivery time,' instead of 'Delivered late.'"  
Wei: "That's precise. The label and tooltip both describe what the calculation counts."  
Me: "Great. So the wording checks out against all 3 shipments."  
Wei: "It does. I'm closing the browser tab now. The spreadsheet can have me back."  

---

## 2026-03-25 15:07 UTC · A counter out of commission (part 2)

`830e4b88-52ac-5c4d-a041-cf690d53d7b5`

**Summary (53 words):** Riley noted the new sink has been working since March 23 and passed testing with a colander. The kitchen is almost finished; a contractor is due March 30 to adjust one cabinet door. A roll of painter's tape remained on the windowsill, but Riley/Wei believed they could handle it themselves, avoiding another contractor.

**Key facts:** The new sink has been working since March 23. | The contractor is due back on March 30 to adjust one cabinet door.

**Minor details:** Wei tested the sink by rinsing a colander. | A roll of painter's tape was left on the windowsill.

Me: "So is the new sink working yet?"  
Wei: "It is. It's been working since March 23."  
Me: "Oh, nice. No more washing dishes in the bathroom?"  
Wei: "That's the idea. I tested the sink by rinsing a colander, which seemed like a suitably rigorous experiment."  
Me: "And the colander survived?"  
Wei: "It did. The sink also passed."  
Me: "Is the kitchen all finished now, then?"  
Wei: "Almost. The contractor is due back on March 30 to adjust 1 cabinet door."  
Me: "Just the 1 door?"  
Wei: "Just 1. A small adjustment."  
Me: "That's a pretty short list."  
Wei: "Well, there's also a roll of painter's tape left on the windowsill. But I think we can handle that ourselves."  
Me: "Good. I was worried you'd need another contractor."  

---
