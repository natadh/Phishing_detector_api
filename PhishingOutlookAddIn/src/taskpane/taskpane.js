/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
 * See LICENSE in the project root for license information.
 */

/* global document, Office */

// Office.onReady((info) => {
//   if (info.host === Office.HostType.Outlook) {
//     document.getElementById("sideload-msg").style.display = "none";
//     document.getElementById("app-body").style.display = "flex";
//     document.getElementById("run").onclick = run;
//   }
// });

// export async function run() {
//   /**
//    * Insert your Outlook code here
//    */

//   const item = Office.context.mailbox.item;
//   let insertAt = document.getElementById("item-subject");
//   let label = document.createElement("b").appendChild(document.createTextNode("Subject: "));
//   insertAt.appendChild(label);
//   insertAt.appendChild(document.createElement("br"));
//   insertAt.appendChild(document.createTextNode(item.subject));
//   insertAt.appendChild(document.createElement("br"));
// }

Office.onReady((info) => {
  if (info.host === Office.HostType.Outlook) {
      document.getElementById("analyzeButton").onclick = analyzeEmail;
  }
});

function analyzeEmail() {
  Office.context.mailbox.item.body.getAsync(Office.CoercionType.Text, (result) => {
      if (result.status === Office.AsyncResultStatus.Succeeded) {
          const emailBody = result.value;
          sendToPhishingAPI(emailBody);
      } else {
          showMessage("Error retrieving email content.");
      }
  });
}

function sendToPhishingAPI(emailBody) {
  const apiUrl = "http://127.0.0.1:8000/predict"; // Update with your actual API URL

  fetch(apiUrl, {
      method: "POST",
      headers: {
          "Content-Type": "application/json",
      },
      body: JSON.stringify({ email_text: emailBody }),
  })
  .then(response => response.json())
  .then(data => {
      if (data.prediction) {
          showMessage(`Prediction: ${data.prediction} (Confidence: ${data.confidence}%)`);
      } else {
          showMessage("Error analyzing email.");
      }
  })
  .catch(error => {
      console.error("API Error:", error);
      showMessage("Failed to connect to the phishing detection API.");
  });
}

function showMessage(message) {
  document.getElementById("output").innerText = message;
}
