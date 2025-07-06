using Amazon.Lambda.Core;
using Amazon.Lambda.APIGatewayEvents;
using System.Text.Json;
using System.Text.Json.Serialization;
using Amazon.DynamoDBv2;
using Amazon.DynamoDBv2.Model;
using System.Globalization;
using Amazon;

[assembly: LambdaSerializer(typeof(Amazon.Lambda.Serialization.SystemTextJson.DefaultLambdaJsonSerializer))]

namespace MLNET.KMeans.InferenceApi
{
    public class Function
    {
        /// <summary>
        /// A Lambda Function Url fronted service for processing incoming POST requests for clustering predictions
        /// </summary>
        /// <param name="request">The event for the Lambda function handler to process.</param>
        /// <param name="context">The ILambdaContext that provides methods for logging and describing the Lambda environment.</param>
        /// <returns></returns>
        public async Task<PredictionResponse> FunctionHandler(APIGatewayHttpApiV2ProxyRequest request, ILambdaContext context)
        {
            // 1. Parse the incoming request JSON into PredictionRequest
            var predictionRequest = JsonSerializer.Deserialize<PredictionRequest>(request.Body);

            using var dynamoClient = new AmazonDynamoDBClient(RegionEndpoint.APSoutheast2);

            var getItemReq = new GetItemRequest
            {
                TableName = "GameClusters",
                Key = new Dictionary<string, AttributeValue>
                {
                    // Assuming app_id is stored as a number
                    ["app_id"] = new AttributeValue { N = predictionRequest.AppId }
                }
            };

            var getItemRes = await dynamoClient.GetItemAsync(getItemReq);

            if (getItemRes.Item == null || getItemRes.Item.Count == 0)
            {
                return new PredictionResponse
                {
                    Results = new List<GameDetail> { new GameDetail { Error = "Record not found." } }
                };
            }

            // 3. Parse the retrieved item's cluster_id and game_json
            var clusterId = int.Parse(getItemRes.Item["cluster_id"].N, CultureInfo.InvariantCulture);
            var distance = float.Parse(getItemRes.Item["distance_to_centroid"].N, CultureInfo.InvariantCulture);
            var imageUrl = getItemRes.Item["image_url"].S;
            var serializedGameJson = getItemRes.Item["game_json"].S;

            // Deserialize the nested game_json object
            var gameData = JsonSerializer.Deserialize<Game.GameData>(serializedGameJson);

            // 4. Scan DynamoDB to find other records in the same cluster, sort by distance
            //    and return only the number specified by SimilarGameCount
            var scanReq = new ScanRequest
            {
                TableName = "GameClusters",
                FilterExpression = "cluster_id = :cid",
                ExpressionAttributeValues = new Dictionary<string, AttributeValue>
                {
                    [":cid"] = new AttributeValue { N = clusterId.ToString(CultureInfo.InvariantCulture) }
                }
            };

            var scanRes = dynamoClient.ScanAsync(scanReq).Result;
            var clusterItems = scanRes.Items
                .Select(i =>
                {
                    var dist = float.Parse(i["distance_to_centroid"].N, CultureInfo.InvariantCulture);
                    var gjson = i["game_json"].S;
                    var gData = JsonSerializer.Deserialize<Game.GameData>(gjson);
                    return new GameDetail
                    {
                        AppId = int.Parse(i["app_id"].N, CultureInfo.InvariantCulture),
                        DistanceToCentroid = dist,
                        ClusterId = int.Parse(i["cluster_id"].N, CultureInfo.InvariantCulture),
                        GameJson = gjson,
                        ParsedGame = gData,
                        ImageUrl = i.ContainsKey("image_url") ? i["image_url"].S : string.Empty
                    };
                })
                .OrderBy(x => x.DistanceToCentroid)
                .Take(predictionRequest.SimilarGameCount)
                .ToList();

            // 5. Build and return the response, including the original item
            return new PredictionResponse
            {
                Results = clusterItems
            };
        }
    }

    public class PredictionRequest
    {
        [JsonPropertyName("app_id")]
        public string AppId { get; set; }

        [JsonPropertyName("similar_game_count")]
        public int SimilarGameCount { get; set; }
    }

    public class PredictionResponse
    {
        [JsonPropertyName("results")]
        public IReadOnlyCollection<GameDetail> Results { get; set; } = new List<GameDetail>();
    }

    public class GameDetail
    {
        public int AppId { get; set; }
        public float DistanceToCentroid { get; set; }
        public int ClusterId { get; set; }
        public string GameJson { get; set; } = string.Empty;
        public Game.GameData ParsedGame { get; set; }
        public string ImageUrl { get; set; } = string.Empty;
        public string Error { get; set; } = string.Empty;
    }

    public class Game
    {
        public class GameData
        {
            [JsonPropertyName("app_id")]
            public int AppId { get; set; }

            [JsonPropertyName("title")]
            public string Title { get; set; } = string.Empty;

            [JsonPropertyName("price")]
            public float Price { get; set; }

            [JsonPropertyName("image_url")]
            public string ImageUrl { get; set; } = string.Empty;
        }
    }
}