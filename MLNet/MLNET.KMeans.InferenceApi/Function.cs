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

        public Function()
            : this(new AmazonDynamoDBClient(RegionEndpoint.APSoutheast2)) { }

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

            // 1) Lookup cluster and target distance
            var lookup = await _repo.GetClusterForAppAsync(appId);
            if (lookup == null)
                return new PredictionResponse { Results = Array.Empty<GameDetail>() };

            var (clusterId, targetDist) = lookup.Value;

            // 2) Query neighbours, ordered by distance_to_centroid ascending
            var neighbours = await _repo.GetNearestNeighboursAsync(
                clusterId,
                appId,
                targetDist,
                takeN);

            return new PredictionResponse { Results = neighbours };
        }
    }

    public class GameClusterRepository
    {
        private readonly IAmazonDynamoDB _db;
        private const string TableName = "GameClusters";
        private const string GsiName = "app_id-index";          // <-- ensure this matches your actual GSI name
        private const string PkCluster = "cluster_id";
        private const string SkDistance = "distance_to_centroid";
        private const string AttrAppId = "app_id";
        private const string AttrGameJson = "game_json";
        private const string AttrImageUrl = "image_url";

        public GameClusterRepository(IAmazonDynamoDB db) => _db = db;

        /// <summary>
        /// Queries the GSI to fetch both cluster_id and the game's own distance_to_centroid.
        /// </summary>
        public async Task<(int clusterId, float distance)?> GetClusterForAppAsync(int appId)
        {
            var req = new QueryRequest
            {
                TableName = TableName,
                IndexName = GsiName,
                KeyConditionExpression = $"{AttrAppId} = :aid",
                ExpressionAttributeValues = new Dictionary<string, AttributeValue>
                {
                    [":aid"] = new AttributeValue { N = appId.ToString() }
                },
                // Only need these columns
                ProjectionExpression = $"{PkCluster}, {SkDistance}",
                Limit = 1
            };

            var res = await _db.QueryAsync(req);
            if (res.Items.Count == 0)
                return null;

            var item = res.Items[0];
            if (!item.ContainsKey(PkCluster) || !item.ContainsKey(SkDistance))
            {
                throw new InvalidOperationException(
                    $"GSI result missing '{PkCluster}' or '{SkDistance}'. Keys: {string.Join(",", item.Keys)}");
            }

            int clusterId = int.Parse(item[PkCluster].N, CultureInfo.InvariantCulture);
            float distance = float.Parse(item[SkDistance].N, CultureInfo.InvariantCulture);
            return (clusterId, distance);
        }

        /// <summary>
        /// Queries the main table for neighbours in the same cluster,
        /// using ExclusiveStartKey to page *after* the target distance.
        /// </summary>
        public async Task<List<GameDetail>> GetNearestNeighboursAsync(
            int clusterId,
            int excludeAppId,
            float startDistance,
            int count)
        {
            var req = new QueryRequest
            {
                TableName = TableName,
                KeyConditionExpression = $"{PkCluster} = :cid",
                ExpressionAttributeValues = new Dictionary<string, AttributeValue>
                {
                    [":cid"] = new AttributeValue { N = clusterId.ToString() }
                },
                ScanIndexForward = true,  // Ascending: lowest distance first
                Limit = count + 1,
                ExclusiveStartKey = new Dictionary<string, AttributeValue>
                {
                    [PkCluster] = new AttributeValue { N = clusterId.ToString() },
                    [SkDistance] = new AttributeValue { N = startDistance.ToString("R", CultureInfo.InvariantCulture) }
                }
            };

            var res = await _db.QueryAsync(req);
            var result = new List<GameDetail>(count);

            foreach (var item in res.Items)
            {
                int candidateId = int.Parse(item[AttrAppId].N, CultureInfo.InvariantCulture);
                if (candidateId == excludeAppId)
                    continue;

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
                    ImageUrl = item.ContainsKey(AttrImageUrl) ? item[AttrImageUrl].S : string.Empty
                });

                if (result.Count == count)
                    break;
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
