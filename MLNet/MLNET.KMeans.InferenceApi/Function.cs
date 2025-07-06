using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Amazon;
using Amazon.DynamoDBv2;
using Amazon.DynamoDBv2.Model;
using Amazon.Lambda.APIGatewayEvents;
using Amazon.Lambda.Core;

[assembly: LambdaSerializer(typeof(Amazon.Lambda.Serialization.SystemTextJson.DefaultLambdaJsonSerializer))]

namespace MLNET.KMeans.InferenceApi
{
    public class Function
    {
        private readonly GameClusterRepository _repo;

        public Function() : this(new AmazonDynamoDBClient(RegionEndpoint.APSoutheast2)) { }

        internal Function(IAmazonDynamoDB dynamoDbClient)
        {
            _repo = new GameClusterRepository(dynamoDbClient);
        }

        public async Task<PredictionResponse> FunctionHandler(
            APIGatewayHttpApiV2ProxyRequest request,
            ILambdaContext context)
        {
            var req = JsonSerializer.Deserialize<PredictionRequest>(request.Body)!;
            var appId = int.Parse(req.AppId, CultureInfo.InvariantCulture);
            var takeN = req.SimilarGameCount;

            var clusterId = (await _repo.GetClusterForAppAsync(appId)).Value;
            var neighbours = await _repo.GetNearestNeighboursAsync(clusterId, appId, takeN);

            return new PredictionResponse { Results = neighbours };
        }
    }

    public class GameClusterRepository
    {
        private readonly IAmazonDynamoDB _db;
        private const string TableName = "GameClusters";
        private const string GsiName = "app_id-index";
        private const string PkCluster = "cluster_id";
        private const string SkDistance = "distance_to_centroid";
        private const string AttrAppId = "app_id";
        private const string AttrGameJson = "game_json";
        private const string AttrImageUrl = "image_url";

        public GameClusterRepository(IAmazonDynamoDB db) => _db = db;

        public async Task<int?> GetClusterForAppAsync(int appId)
        {
            var q = new QueryRequest
            {
                TableName = TableName,
                IndexName = GsiName,
                KeyConditionExpression = $"{AttrAppId} = :aid",
                ExpressionAttributeValues =
                    new Dictionary<string, AttributeValue>
                    {
                        [":aid"] = new AttributeValue { N = appId.ToString() }
                    },
                Limit = 1
            };
            var res = await _db.QueryAsync(q);
            if (res.Items.Count == 0) return null;
            return int.Parse(res.Items[0][PkCluster].N, CultureInfo.InvariantCulture);
        }

        public async Task<List<GameDetail>> GetNearestNeighboursAsync(
            int clusterId,
            int excludeAppId,
            int count)
        {
            var q = new QueryRequest
            {
                TableName = TableName,
                KeyConditionExpression = $"{PkCluster} = :cid",
                ExpressionAttributeValues =
                    new Dictionary<string, AttributeValue>
                    {
                        [":cid"] = new AttributeValue { N = clusterId.ToString() }
                    },
                ScanIndexForward = true,
                Limit = count + 1
            };

            var res = await _db.QueryAsync(q);
            var result = new List<GameDetail>(count);

            foreach (var item in res.Items)
            {
                var candidateId = int.Parse(item[AttrAppId].N, CultureInfo.InvariantCulture);
                if (candidateId == excludeAppId) continue;

                // Parse the stored JSON into our strong type
                var gData = JsonSerializer.Deserialize<Game.GameData>(
                    item[AttrGameJson].S,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;

                result.Add(new GameDetail
                {
                    AppId = candidateId,
                    ClusterId = clusterId,
                    DistanceToCentroid = float.Parse(item[SkDistance].N, CultureInfo.InvariantCulture),
                    Title = gData.Title,
                    Price = gData.Price,
                    ImageUrl = item.ContainsKey(AttrImageUrl)
                                          ? item[AttrImageUrl].S
                                          : string.Empty
                });

                if (result.Count == count) break;
            }

            return result;
        }
    }

    public class PredictionRequest
    {
        [JsonPropertyName("app_id")]
        public string AppId { get; set; } = default!;

        [JsonPropertyName("similar_game_count")]
        public int SimilarGameCount { get; set; }
    }

    public class PredictionResponse
    {
        [JsonPropertyName("results")]
        public IReadOnlyCollection<GameDetail> Results { get; init; } = Array.Empty<GameDetail>();
    }

    public class GameDetail
    {
        public int AppId { get; set; }
        public int ClusterId { get; set; }
        public float DistanceToCentroid { get; set; }
        public string Title { get; set; } = string.Empty;
        public float Price { get; set; }
        public string ImageUrl { get; set; } = string.Empty;
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
