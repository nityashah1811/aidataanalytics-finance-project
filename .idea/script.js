async function callModel() {
    const input = document.getElementById("userInput").value;
    const resultDiv = document.getElementById("result");

    resultDiv.innerHTML = "Loading...";

    try {
        const response = await fetch("https://your-ml-api.com/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: input })
        });

        const data = await response.json();
        resultDiv.innerHTML = "Prediction: " + data.prediction;

    } catch (error) {
        resultDiv.innerHTML = "Error calling model.";
        console.error(error);
    }
}
