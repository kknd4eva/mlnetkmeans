# MLNet KMeans Inference API

This project provides a serverless API for querying game cluster recommendations based on ML.NET’s KMeans model. The API exposes an endpoint that retrieves a game record from DynamoDB and returns similar games in the same cluster, sorted by the closest distance to centroid.

## Special Thanks
- Kaggle.com for various Steam game datasets (https://www.kaggle.com/search?q=steam+in%3Adatasets)

## Project Purpose

- **Clustering**: Uses ML.NET to group games into clusters.
- **Inference**: Exposes a lightweight inference endpoint to retrieve recommendations by app ID.
- **Serverless**: Hosted as an AWS Lambda function.

## Setup Steps

1. **Run the MLNet KMeans Project**  
   - Generate and save the ML.NET model.  
   - As part of the execution, the code will create a DynamoDb table named `GameClusters` as follows: 
        -- table schema (partition key: `cluster__id`, sort key: `distance_to_centroid`).
        -- Create a Global Secondary Index (GSI) on `app_id` for obtaining a single game record.
        -- Provide sufficient read/write throughput or use on-demand capacity.
   - Export data to populate your DynamoDB table (if desired) using the provided functionality in the KMeans project.

2. **Deploy the Inference API Using SAM**  
   - From the root or relevant folder, run:  
     ```bash
     sam deploy --template serverless.yaml --stack-name mlnetkmeans --resolve-s3 --capabilities "CAPABILITY_IAM"
     ```
   - This will package and deploy the Lambda function, creating one Lambda URL for inference queries.

Once deployed, you can invoke the endpoint with a POST request supplying the required JSON payload (app_id, similar_game_count). The function will fetch the matching record, find other games in the same cluster, and return a list sorted by ascending distance to centroid.

Here's an example payload and response: 

Request:
```
{
    "app_id": "925670",
    "similar_game_count": 5
}
```
Response:
```
{
    "results": [
        {
            "AppId": 348950,
            "ClusterId": 10,
            "Price": 9.99,
            "DistanceToCentroid": 5.8485,
            "Title": "Ame no Marginal -Rain Marginal-",
            "ImageUrl": "https://steamcdn-a.akamaihd.net/steam/apps/348950/header.jpg?t=1547764457"
        },
        {
            "AppId": 869870,
            "ClusterId": 10,
            "Price": 8.29,
            "DistanceToCentroid": 6.1177,
            "Title": "She is Mermaid",
            "ImageUrl": "https://steamcdn-a.akamaihd.net/steam/apps/869870/header.jpg?t=1549632599"
        },
        {
            "AppId": 896500,
            "ClusterId": 10,
            "Price": 2.09,
            "DistanceToCentroid": 6.3039,
            "Title": "Whispering Flames",
            "ImageUrl": "https://steamcdn-a.akamaihd.net/steam/apps/896500/header.jpg?t=1542546604"
        },
        {
            "AppId": 770600,
            "ClusterId": 10,
            "Price": 1.69,
            "DistanceToCentroid": 6.321,
            "Title": "Memento of Spring",
            "ImageUrl": "https://steamcdn-a.akamaihd.net/steam/apps/770600/header.jpg?t=1515513139"
        },
        {
            "AppId": 407320,
            "ClusterId": 10,
            "Price": 6.99,
            "DistanceToCentroid": 6.3226,
            "Title": "My Little Kitties",
            "ImageUrl": "https://steamcdn-a.akamaihd.net/steam/apps/407320/header.jpg?t=1547764524"
        }
    ]
}
```