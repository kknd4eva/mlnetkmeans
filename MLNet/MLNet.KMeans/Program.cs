using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
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
            // 1) Deserialize request (assumes well-formed JSON)
            var req = JsonSerializer.Deserialize<PredictionRequest>(request.Body)!;
            var appId = int.Parse(req.AppId, CultureInfo.InvariantCulture);
            var takeN = req.SimilarGameCount;

            // 2) Lookup cluster_id for this app
            var clusterId = (await _repo.GetClusterForAppAsync(appId)).Value;

            // 3) Query the N nearest neighbours (skipping the original app)
            var neighbours = await _repo.GetNearestNeighboursAsync(
                clusterId,
                appId,
                takeN);

            // 4) Return directly
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
                ExpressionAttributeValues = new Dictionary<string, AttributeValue>
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
                ExpressionAttributeValues = new Dictionary<string, AttributeValue>
                {
                    [":cid"] = new AttributeValue { N = clusterId.ToString() }
                },
                ScanIndexForward = true,    // sort by distance asc
                Limit = count + 1
            };

            var res = await _db.QueryAsync(q);
            var result = new List<GameDetail>(count);

            foreach (var item in res.Items)
            {
                var id = int.Parse(item[AttrAppId].N, CultureInfo.InvariantCulture);
                if (id == excludeAppId) continue;

                result.Add(new GameDetail
                {
                    AppId = id,
                    ClusterId = clusterId,
                    DistanceToCentroid = float.Parse(item[SkDistance].N, CultureInfo.InvariantCulture),
                    GameJson = item[AttrGameJson].S,
                    ParsedGame = JsonSerializer.Deserialize<Game.GameData>(item[AttrGameJson].S)!,
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
        public string GameJson { get; set; } = string.Empty;
        public Game.GameData ParsedGame { get; set; } = default!;
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
